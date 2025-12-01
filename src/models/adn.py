import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .base import Base, BaseTrain
from ..networks import ADN, NLayerDiscriminator, add_gan_loss, ADN_VAE
from ..utils import print_model, get_device


class ADNTrain(BaseTrain):
    """
    Training wrapper for ADN generator and its corresponding discriminators.

    This class handles:
    - Forward passes for all ADN submodules
    - All loss computations (L1, GAN, consistency losses, artifact loss)
    - Mixed precision scaling
    - Generator and discriminator optimization steps
    - Tensor cleanup to reduce VRAM usage
    """

    def __init__(self, learn_opts, loss_opts, g_type, d_type, **model_opts):
        super(ADNTrain, self).__init__(learn_opts, loss_opts)

        g_opts, d_opts = model_opts[g_type], model_opts[d_type]

        model_dict = dict(
            adn=lambda: ADN(**g_opts),
            nlayer=lambda: NLayerDiscriminator(**d_opts)
        )

        # ADN generator
        self.model_g = self._get_trainer(model_dict, g_type)

        # Discriminators (LQ and HQ)
        self.model_dl = add_gan_loss(self._get_trainer(model_dict, d_type))  # LQ discriminator
        self.model_dh = add_gan_loss(self._get_trainer(model_dict, d_type))  # HQ discriminator

        # Registered losses
        loss_dict = dict(
            l1=nn.L1Loss,
            gl=(self.model_dl.get_g_loss, self.model_dl.get_d_loss),  # GAN loss (LQ)
            gh=(self.model_dh.get_g_loss, self.model_dh.get_d_loss)   # GAN loss (HQ)
        )

        # Attach loss criteria with configured weights
        self.model_g._criterion["ll"] = self._get_criterion(loss_dict, self.wgts["ll"], "ll_")
        self.model_g._criterion["lh"] = self._get_criterion(loss_dict, self.wgts["lh"], "lh_")
        self.model_g._criterion["hh"] = self._get_criterion(loss_dict, self.wgts["hh"], "hh_")
        self.model_g._criterion["lhl"] = self._get_criterion(loss_dict, self.wgts["lhl"], "lhl_")
        self.model_g._criterion["hlh"] = self._get_criterion(loss_dict, self.wgts["hlh"], "hlh_")
        self.model_g._criterion["art"] = self._get_criterion(loss_dict, self.wgts["art"], "art_")
        self.model_g._criterion["gl"] = self._get_criterion(loss_dict, self.wgts["gl"])
        self.model_g._criterion["gh"] = self._get_criterion(loss_dict, self.wgts["gh"])

        print_model(self)

    def _nonzero_weight(self, *names):
        """Utility: return the sum of configured loss weights for given keys."""
        wgt = 0
        for name in names:
            w = self.wgts[name]
            if isinstance(w[0], str):
                w = [w]
            for p in w:
                wgt += p[1]
        return wgt

    def optimize(self, img_low, img_high, scaler):
        """Runs a full optimization step: forward → losses → backward → updates."""
        self.img_low, self.img_high = self._match_device(img_low, img_high)

        # --- Reset gradient histories ---
        self.model_g._clear()
        self.model_dl._clear()
        self.model_dh._clear()

        # ============================================================
        #                       FORWARD + LOSSES
        # ============================================================

        # low → (ll, lh)
        if self._nonzero_weight("gl", "lh", "ll"):
            self.pred_ll, self.pred_lh = self.model_g.forward1(self.img_low)
            self.model_g._criterion["gl"](self.pred_lh, self.img_high)
            self.model_g._criterion["lh"](self.pred_lh, self.img_high)
            self.model_g._criterion["ll"](self.pred_ll, self.img_low)

        # high → (hl, hh)
        if self._nonzero_weight("gh", "hh"):
            if not hasattr(self, "pred_lh"):
                _, self.pred_lh = self.model_g.forward1(self.img_low)
            self.pred_hl, self.pred_hh = self.model_g.forward2(self.img_low, self.img_high)
            self.model_g._criterion["gh"](self.pred_hl, self.img_low)
            self.model_g._criterion["hh"](self.pred_hh, self.img_high)

        # low_h → lhl
        if self._nonzero_weight("lhl"):
            if not hasattr(self, "pred_hl"):
                self.pred_hl, _ = self.model_g.forward2(self.img_low, self.img_high)
            if not hasattr(self, "pred_lh"):
                _, self.pred_lh = self.model_g.forward1(self.img_low)

            self.pred_lhl = self.model_g.forward_hl(
                self.pred_hl.detach(), self.pred_lh.detach())
            self.model_g._criterion["lhl"](self.pred_lhl, self.img_low)

        # high_l → hlh
        if self._nonzero_weight("hlh"):
            if not hasattr(self, "pred_hl"):
                self.pred_hl, _ = self.model_g.forward2(self.img_low, self.img_high)

            self.pred_hlh = self.model_g.forward_lh(self.pred_hl.detach())
            self.model_g._criterion["hlh"](self.pred_hlh, self.img_high)

        # Artifact constraint
        if self._nonzero_weight("art"):
            if not hasattr(self, "pred_ll"):
                self.pred_ll, self.pred_lh = self.model_g.forward1(self.img_low)
            if not hasattr(self, "pred_hh"):
                _, self.pred_hh = self.model_g.forward2(self.img_low, self.img_high)

            ll = self.img_low if self.gt_art else self.pred_ll
            hh = self.img_high if self.gt_art else self.pred_hh

            self.model_g._criterion["art"](ll - self.pred_lh, self.pred_hl - hh)

        # ============================================================
        #                         OPTIMIZATION
        # ============================================================

        # 1. Update generator
        self.model_g._update(scaler=scaler)

        # 2. Update LQ and HQ discriminators
        if self._nonzero_weight("gl"):
            self.model_dl._update(scaler=scaler)
        if self._nonzero_weight("gh"):
            self.model_dh._update(scaler=scaler)

        # 3. Update mixed precision scaler
        if scaler:
            scaler.update()

        # Cleanup
        for attr in ("pred_ll", "pred_lh", "pred_hl", "pred_hh", "pred_lhl", "pred_hlh"):
            if hasattr(self, attr):
                delattr(self, attr)

        # Merge losses for reporting
        self.loss = self._merge_loss(
            self.model_dl._loss,
            self.model_dh._loss,
            self.model_g._loss
        )

    def get_visuals(self, n=8):
        lookup = [
            ("l", "img_low"), ("ll", "pred_ll"), ("lh", "pred_lh"), ("lhl", "pred_lhl"),
            ("h", "img_high"), ("hl", "pred_hl"), ("hh", "pred_hh"), ("hlh", "pred_hlh")
        ]
        return self._get_visuals(lookup, n)

    def evaluate(self, loader, metrics):
        """Evaluate ADN performance on a dataset using any metric function."""
        progress = tqdm(loader)
        res = defaultdict(lambda: defaultdict(float))
        cnt = 0

        for img_low, img_high in progress:
            img_low, img_high = self._match_device(img_low, img_high)

            def to_numpy(*data):
                data = [loader.dataset.to_numpy(d, False) for d in data]
                return data[0] if len(data) == 1 else data

            pred_ll, pred_lh = self.model_g.forward1(img_low)
            pred_hl, pred_hh = self.model_g.forward2(img_low, img_high)
            pred_hlh = self.model_g.forward_lh(pred_hl)

            img_low, img_high, pred_ll, pred_lh, pred_hl, pred_hh, pred_hlh = to_numpy(
                img_low, img_high, pred_ll, pred_lh, pred_hl, pred_hh, pred_hlh
            )

            met = dict(
                ll=metrics(img_low, pred_ll),
                lh=metrics(img_high, pred_lh),
                hl=metrics(img_low, pred_hl),
                hh=metrics(img_high, pred_hh),
                hlh=metrics(img_high, pred_hlh)
            )

            res = {n: {k: (res[n][k] * cnt + v) / (cnt + 1) for k, v in met[n].items()} for n in met}
            cnt += 1

            desc = "[{}]".format("/".join(met["ll"].keys()))
            for n, m in res.items():
                vals = "/".join("{:.2f}".format(v) for v in m.values())
                desc += f" {n}: {vals}"

            progress.set_description(desc=desc)


class ADNTest(Base):
    """
    Inference wrapper for ADN. Handles:
    - Forward passes
    - Evaluation
    - Paired output extraction for visualization
    """

    def __init__(self, g_type, **model_opts):
        super(ADNTest, self).__init__()

        g_opts = model_opts[g_type]
        model_dict = dict(adn=lambda: ADN(**g_opts))
        self.model_g = model_dict[g_type]()

        print_model(self)

    def forward(self, img_low):
        self.img_low = self._match_device(img_low)
        self.pred_ll, self.pred_lh = self.model_g.forward1(self.img_low)
        return self.pred_ll, self.pred_lh

    def evaluate(self, img_low, img_high, name=None):
        self.img_low, self.img_high = self._match_device(img_low, img_high)
        self.name = name

        self.pred_ll, self.pred_lh = self.model_g.forward1(self.img_low)
        self.pred_hl, self.pred_hh = self.model_g.forward2(self.img_low, self.img_high)
        self.pred_hlh = self.model_g.forward_lh(self.pred_hl)

    def get_pairs(self):
        return [
            ("before", (self.img_low, self.img_high)),
            ("after", (self.pred_lh, self.img_high))
        ], self.name

    def get_visuals(self, n=8):
        lookup = [
            ("l", "img_low"), ("ll", "pred_ll"), ("lh", "pred_lh"),
            ("h", "img_high"), ("hl", "pred_hl"), ("hh", "pred_hh")
        ]
        func = lambda x: x * 0.5 + 0.5
        return self._get_visuals(lookup, n, func, False)


class ADN_VAETrain(BaseTrain):
    def __init__(self, learn_opts, loss_opts, g_type, d_type, **model_opts):
        super(ADN_VAETrain, self).__init__(learn_opts, loss_opts)
        g_opts, d_opts = model_opts[g_type], model_opts[d_type]
        
        # Init with VAE model
        model_dict = dict(
            adn = lambda: ADN_VAE(**g_opts),
            nlayer = lambda: NLayerDiscriminator(**d_opts)
        )
        
        # Generator
        self.model_g = self._get_trainer(model_dict, g_type)
        
        # Discriminators (Low & High)
        self.model_dl = add_gan_loss(self._get_trainer(model_dict, d_type))
        self.model_dh = add_gan_loss(self._get_trainer(model_dict, d_type))
        
        loss_dict = dict(
            l1 = nn.L1Loss,
            kld = KLDLoss,
            gl = (self.model_dl.get_g_loss, self.model_dl.get_d_loss),
            gh = (self.model_dh.get_g_loss, self.model_dh.get_d_loss),
        )
        
        # Create criterion
        self.model_g._criterion["ll"] = self._get_criterion(loss_dict, self.wgts["ll"], "ll_")
        self.model_g._criterion["lh"] = self._get_criterion(loss_dict, self.wgts["lh"], "lh_")
        self.model_g._criterion["hh"] = self._get_criterion(loss_dict, self.wgts["hh"], "hh_")
        self.model_g._criterion["lhl"] = self._get_criterion(loss_dict, self.wgts["lhl"], "lhl_")
        
        self.model_g._criterion["hlh"] = self._get_criterion(loss_dict, self.wgts["hlh"], "hlh_")
        self.model_g._criterion["art"] = self._get_criterion(loss_dict, self.wgts["art"], "art_")
        self.model_g._criterion["gl"] = self._get_criterion(loss_dict, self.wgts["gl"])
        self.model_g._criterion["gh"] = self._get_criterion(loss_dict, self.wgts["gh"])
        
        self.model_g._criterion["kld"] = self._get_criterion(loss_dict, self.wgts.get("kld", [None, 0.0]), "kld_")
        
        print_model(self)
        
    def optimize(self, img_low, img_high, scaler=None):
        self.img_low, self.img_high = self._match_device(img_low, img_high)
        
        self.model_g._clear()
        self.model_dl._clear()
        self.model_dh._clear()
        
        # --- 1. Low Quality Reconstruction & GAN ---
        if self._nonzero_weight("gl", "lh", "ll", "kld"):
            self.pred_ll, self.pred_lh = self.model_g.forward1(self.img_low)
            
            self.model_g._criterion["gl"](self.pred_lh, self.img_high)
            self.model_g._criterion["lh"](self.pred_lh, self.img_high)
            self.model_g._criterion["ll"](self.pred_ll, self.img_low)
            
            if hasattr(self.model_g, "mu_low") and hasattr(self.model_g, "logvar_low"):
                self.model_g._criterion["kld"](self.model_g.mu_low, self.model_g.logvar_low)
                
        # --- 2. High Quality Reconstruction & GAN ---
        if self._nonzero_weight("gh", "hh", "kld"):
            if not hasattr(self, "pred_lh"):
                _, self.pred_lh = self.model_g.forward1(self.img_low)
                
            self.pred_hl, self.pred_hh = self.model_g.forward2(self.img_low, self.img_high)
            
            self.model_g._criterion["gh"](self.pred_hl, self.img_low)
            self.model_g._criterion["hh"](self.pred_hh, self.img_high)
            
            if hasattr(self.model_g, "mu_high") and hasattr(self.model_g, "logvar_high"):
                self.model_g._criterion["kld"](self.model_g.mu_high, self.model_g.logvar_high)
        
        # --- 3. Cycle Consistency ---
        if self._nonzero_weight("lhl"):
            if not hasattr(self, 'pred_hl'): self.pred_hl, _ = self.model_g.forward2(self.img_low, self.img_high)
            if not hasattr(self, 'pred_lh'): _ , self.pred_lh = self.model_g.forward1(self.img_low)
            self.pred_lhl = self.model_g.forward_hl(self.pred_hl.detach(), self.pred_lh.detach())
            self.model_g._criterion["lhl"](self.pred_lhl, self.img_low)
            
        if self._nonzero_weight("hlh"):
            if not hasattr(self, 'pred_hl'): self.pred_hl, _ = self.model_g.forward2(self.img_low, self.img_high)
            self.pred_hlh = self.model_g.forward_lh(self.pred_hl.detach())
            self.model_g._criterion["hlh"](self.pred_hlh, self.img_high)

        # --- 4. Artifact Consistency ---
        if self._nonzero_weight("art"):
             if not hasattr(self, 'pred_ll'): self.pred_ll, self.pred_lh = self.model_g.forward1(self.img_low)
             if not hasattr(self, 'pred_hl'): self.pred_hl, _ = self.model_g.forward2(self.img_low, self.img_high)
             
             ll = self.img_low
             hh = self.img_high
             self.model_g._criterion["art"](ll - self.pred_lh, self.pred_hl - hh)
            
        # Update weights
        self.model_g._update(scaler=scaler)

        if self._nonzero_weight("gl"):
            self.model_dl._update(scaler=scaler)

        if self._nonzero_weight("gh"):
            self.model_dh._update(scaler=scaler)

        if scaler: scaler.update()

        # Cleanup
        if hasattr(self, 'pred_ll'): del self.pred_ll
        if hasattr(self, 'pred_lh'): del self.pred_lh
        if hasattr(self, 'pred_hl'): del self.pred_hl
        if hasattr(self, 'pred_hh'): del self.pred_hh
        if hasattr(self, 'pred_lhl'): del self.pred_lhl
        if hasattr(self, 'pred_hlh'): del self.pred_hlh

        self.loss = self._merge_loss(
            self.model_dl._loss, self.model_dh._loss, self.model_g._loss)

    def get_visuals(self, n=8):
        lookup = [
            ("l", "img_low"), ("ll", "pred_ll"), ("lh", "pred_lh"), ("lhl", "pred_lhl"),
            ("h", "img_high"), ("hl", "pred_hl"), ("hh", "pred_hh"), ("hlh", "pred_hlh")]

        return self._get_visuals(lookup, n)



class ADN_ArtifactGenerator(Base):
    def __init__(self, g_type, **model_opts):
        super(ADN_ArtifactGenerator, self).__init__()
        g_opts = model_opts[g_type]
           
        model_dict = dict(adn=lambda: ADN_VAE(**g_opts))
        self.model_g = model_dict[g_type]()

        print_model(self)

        self.eval() 

    def generate_artifacts(self, img_clean, img_reference_artifact):
        self.img_clean, self.img_ref = self._match_device(img_clean, img_reference_artifact)
        
        with torch.no_grad():
            
            self.generated_image = self.model_g.forward_hl(
                x_low=self.img_ref,
                x_high=self.img_clean
            )
            
        return self.generated_image

    def get_visuals(self, n=8):

        lookup = []
        if hasattr(self, 'img_clean'): lookup.append(("1_Clean_Input", "img_clean"))
        if hasattr(self, 'img_ref'): lookup.append(("2_Artifact_Ref", "img_ref"))
        if hasattr(self, 'generated_image'): lookup.append(("3_Generated_Output", "generated_image"))

        func = lambda x: x * 0.5 + 0.5 
        
        return self._get_visuals(lookup, n, func, False)