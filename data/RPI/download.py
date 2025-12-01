import os
import requests
import tarfile
import argparse

from tqdm import tqdm
import sys
import random  # Added for splitting logic

RANDOM_SEED = 44

URLS = {
    "head": [
        # Head01
        "https://public.boxcloud.com/d/1/b1!139hGifu6162dKKMjch7nvtqcDHjQpViV5DXUbxfu61tNVO1NYyj-NvIfs_yyLno_ZSkCKrDl4zbK1lTxpV9j-X2CyJBF5QGEzr9DUYP9f3NVGbtfkHtQr1EYTQPDkmim3OWb5ezLgTJoANotnQHbQJVQvG-A-xYHqNkeAEim2F4rPiVskSG7dGLSv2R7buuQTQ1F0WbngP3MDffQB2Jyhx3tANcrmWneNCK_pC4JhPZA_TlEyJevSzReQ_30unzglGfTTyunVq5J5gskTZNk3ku663i5IIHk2j6Nzpk407wfFRPLd75ERQW7zO0KgWZ2r0NgWwi-g6vYyaQeISJEQf29BLSfgy39vEqPk9wC2uUMtVdcx8N3pbXG-m00d-2NyQTfwEyIH_Sgp1htiyNxb8xcnOWKW_4nLR5D5mudivzTSyQWQDlS1dRZoJF5dFRYxmM5t9G8EYQKp94hPRwwGCA_eSDW43xFZr-pzVQLgxaHaO8GCoolzKOrARXsiCzHQJnJmfTuLqderEPoSaQ9ObxcQck6vOAb2QhqmCAaWRJKhjBc0QIoj4N750KQsIF0uSPO_FhMyfsFequ6tVgBGp0p3OaiEeNE4zdMkU6H7EzkbYIXyurIBYPDTq1TpSs4BpHXjBRwJc_WufFyYXOQX2FzlWR8NM1lg-0Rqge0O8ASSaz-Ls8FAMhU3ogXWxYWiSQSqrDmxBg02qVo7HT0K8_XIvqA4H8psXRDorA1g_T7SDB_xfEnRKLoPvo2CudMsiKsVhdALr_L6YN8fu8vr3YCJtpjhoVPl8tUDMMFC4kgqTjViKg2zGQ1VbDuXKiWLr3rsiBl5a1CYvEQSaIi0Zgy4Wqt4omx9ViKZYYFs2-Iy97-9pONRj9VzBXUm3HDG5RRFke7JQQb_Bkrpx2zs7i-bRDsEpvgxh_6A2h6jW4JLwub6RCE4HV-_tAEwxZdhDNUytauvqnMBqIBL-rFWot_jxdY_XHPOEx18R7auEaC7gRkbr4Q9Zi3gl70PX3IjEQoseioD1oPjGjPhsDncGX36twnSvloTF3wxyi5hZ7DSegnTMJ6MfxHG5y0vNjzMpg3W_WKmkCblfFoCvWaWWhd0JfTjTYwaVTs0BgZA0QCQwN9sWWArUidg-hFA7IuCECWrSv2yzu9RpKUAX8BuqxW2hXfW-d02YWaLd-VnKYiRY5b_pEez0VzhwUW4OwfKCbeBT2zFGXrpc4wTmpb8eV9WIWjazTx8VW20QQjyUTsKX_2VGyZsarzc225YZ4UN0kG-JGlQoZGsVMcdphTcm1Umz5VcrXHeEbw9MmC85xKEnRWNv_YhwgR2wemk5JXgkTqia8nZLYXLTrMQ7yeg7Rg0lU1UFltoAl1ARVQRJAbCAqBfuRU2RutH35VCAnSNaEhi60oNWikSE89bJ-MYyR/download",
        # Head01
        "https://public.boxcloud.com/d/1/b1!yXuNSg7qvlEEcExXKVuMncj9BrE1uUWHI-oTZBWRs7MHgxJ_t9oum5W1fHouG05GarZVX_qClfWV-fCH3b4okMWSg0iCVFCNS3O-BR0UNPBjzzcJ3dHB5s6Vwtl6stERhvSBYoQRxNis2TLPaQkcwJdcuhQGLAkXVDXg-vuLjSNco2ZAQppFHtrjt2smhH0VZ9uzP3-d7XbHTaq1ocX-02OyjPkYGJuvqc4gfzJ0G00sElZ0VkKBoJ0JecmTd0KJmAv8I1VaQFlg7n4Lr5tomAopdl8HQErjs_ehwz_NebSvPLrAXE6VuPW7qHnlTiX6bnyk5LQ3wclDN1rLNWAl7WV6CeloskBrNQEpT7QcWeX1Uk8wrrb2oeJP8mpcTPYiaR7yGr0Z9AgOp7Bot5bBz73sYMgpj8oFQTca5F2o2YAd3me1Kwl762xnslLF72_Xaify-xfte9pAWJU9h1fqbCnumytEvSxWyd_PiWQlE4UE_cUBdQQcyjGfcYlMYLE8rh47N9IMkDwuyCablVPHKPlo4I_G9kdzETe-bhXmFgPcNXHhQMPtS-ksiMXDuRBpMpUl8GboTTnyXyp_kJUOlscQC9JQxPmXoKcS6_LVO9ncauevbbrrNTpdTSkYI0r97-BgZ69vFYXcdBmv8ErxTCL-UHl2VDKODudntSGMhDeLruwimZhV8ggGzOFyvi4_9ekCo6g_ftmNt7oGnHj1HZ3E813eXB2ozeMGUohaTxT_9IgnSS8fTDQZWPmvCkNuglGFlji4ED4Mdun3jZAMKyaJsYgAaBrN0TmIstOK2H9zo3F1k9QQgmAXR_dGB_CgVVG5tv0Rp-C3C1Z-fyH1O4BEFcX0DS2Dh5x4qn9k33xQRwPXvQzR94dQP9wRF_IyL_R0rDcchGIXGxlKpBsIaeHkDRtrPMOQ5J4wdwMPpV3CC-v7VbaWv0m-xQjaK0QCw4kgK1fOnP2W5Rq4CVBkC4Yg9CWZ6kFrjqLw2TVBD5z7G2NyFXKmoTkqCzxLf9HzA4jhxCt-4jGvfJ4rXHSK6t17Tmpq3ET2C928xgQnl7f_An7W-B8KYfqES_JqrzRzZINp2BuiDHYWgy-CzM13GHzvHvXc_wgBjIKNCMJyYMC97WZT27Ao6qQT6iLmov-zWffLUzTwlHNBmxL4vQuv5bG7LPlTsKHyabXeOZk-CbAstymmrM_FP3jP8gpIyJmzA2f2L5eBw1YqPewvlNwXeR-H79R-B-OpYZbaUYPZPY8h2Z26oOoO9XWU3b1rkqdM2hJH55_nl9GHO013cUlU4z-cn7EeJW2z16ASN26S6nkv08lcOXxtGq6C-VrXomH4VV36hS_CdliFl_I2w0FL3VHNCLZ2r3ZIBlFtV8L1FUcE1Xqi07aGeEQZc6FCEWqmrd8dAO2r40AmDCkfHuuUylqj/download"
    ],
    "body": [
        # Body01
        "https://public.boxcloud.com/d/1/b1!REt-lKdOKE1z2VK1Scr0p_PfidwQ29CkDNKZsA38vxcAL7MmOolhfzQON0ogoQ1iPKAgLFr-FRJVryt1X_G91cCwfUUdo-deFoDGY58T6RmILPUF3xJXs5BM5Y-0Z9tZfalNEu40pzzfq2Vn_ndivuZ3FjkWFqwMUT-4ACHJiR6CXko1bu2d-UhJjszE_JrNklY7onN5y_G4CnPSNjJsNlBt4onW7afhuAA1-hTPw_BFxDuzKnsnO58G5F11aljQcPmNxdfg9rQFvzsIIwOOih_UUd2MlWicCqskvuc_FYFee_kHvl20n8p4ltP4Y8EbzANSqRk7p0h6wQY9HmczVajhj-zTLzeE7PjXr-OZLzVemfEUmOUpO5oi2-tis5bxhaejaz6Cl5YIgVJzOa7S2NrbpeYcfbv020TTPbVIlMGOvYMrTixfuK7ZPmZ4u-nAVZ4wlnT0_QUhNn0M_67z_qs_afvqm0FbIiimWZRbqnYl1Zo2o9VPP3qbWBFGD5Q7jPwvPbwlRKogGfSqTs2YYszm3JYVfIO3g9osRNpDk9CotIbx3VhVBBD5Jtt1BGIDwHLsewaCd2_JOcSTW62-bGifGds2jrVuEzv8nw83NjnQUwsv105rer08mIJRPMvYoeP44cpQSOeuT2FujCrfEr_vnglIOXCVhDbWxWNGex-ad0ahTMp5jzV2xzASiKgcszWPC75eDz-MT4FsDdBlP3obdCmF9SlhGs-5S5ApbEVcZQFhUmVsSQCEraHY2bdwRZYySiF87yaGn2CCvFwHZwhsCcCwcmQNNHoaR20AiDkI4Bs22ANRC6zvXfvmktIufikN1-wInrhptPAQhY-tE5i5sjOBRTxvAaxCzCf1TkpHL3us-VtMUEMrgC9A0KJcygV6iYTsKlZKZBCKRA20z73OsV8j1Q_-JPa_BC-lN0XKIJtoIrg667rqPyIu_n7fGgyKYa9SJKsWuTadDbJ5R_2iGI_hwbFmm9peCPZwGCuc0KIkVttsQxn-apHGmdjONbuTHaRbj1_inAZGtLSp4Yml0d2n6o23OOuROGazcjXW1rVT6R7azTG5SpLP5NotE2iZUufp49yAKqoo2O8_P2EnJg-ROuLtUmjHLAgabGJ16gIaXY8Tni45w8leYRc3w5CNxdgx9zkePsHy8LocFGdikm1iZHnDYYR58Gf9NbzkSpcQE-T3gJ6H4FhXbCjVvUcdTGHrZqN4K8-BtwIK99cHO3xBzSNVu_C534eDzznZlAvgtFW2-RD95OPoOCb8rwSDivRSIegpWvqTCKCEaeizofD8rUbCXProaSp8CLtoV5taZ9la8dLzoteHeoT_nvkr2QGGw_gsP0R02VSNlj11OSu-4faiW-Sf2__zcMFaL16U_7dnHAa08wgSn2NNGXAVLvFuQkzObgCk89wm75Gq/download",
        # Body02
        "https://public.boxcloud.com/d/1/b1!MtdBpxGQyEUfxW4ZcjZIFefCXOaTnqY75hMH1ZsMBYejS4mzZxWytF1dPTw-NXxYRL_2-29nMMnH5_lmQVznCZZTtb3Eva1huoxcPKSvsFCZuyKtquKwrPWlNaHpaLEHzyVbKieqKFahPjNJB9UQcUMCPDyPHCW7QLbT-qa-Oifbd4DuEuI3NxT30yvmkQrLuQHz_CipQu-eWZwcel-kyxads2rCAltXb_2vmlLBvXqSNMkisHRzeLWWprMTsQGCTLjFxq6QCSs-cs-dySh2a0ejDl9aXcCkBMLy-FpL3hMZrxtBeWDQnOiRXwHwd8GrVUHLY5jJgx6imTeN269lLrGQCZ-g7SwvPC7RcqloE03faDXdhbVhDHDFYJ4CzGo6mR10sgGHv8qAELX93XeYOx_i6Tg_m1jeK0I2HJIPdhla9DEgKNIP1luY3swW_DrsYn49JbIMVrpbTujSV5D1UC26SSuN80DVaKrJon_wX2RYtR_DaHAMf7ta7q6rqXLOv4MYWMSQFw0gVxIvcpbNI-azyuA9KG9ZUaFARCpv-OBzA4_PEi-Q5ueLYkeYXcwfmxsOF2wiojGlOMK6d22fpOFDyskkBnY6gfkdJrqTdY29nl7J3vQFwcxdaue-NZLha7NaCbV8xSyIfNPvvA4wtveHAMEZeKggsPmf0NxguzVeZ2hpmcoyg40_QFL-NfZsox8Z3UWFn209Y8Dhs4SLY7fUDJ7VOH3sE6BuH3gGMGu5w-pL0JF_yU7IBhtY2aS2cYbJMzMer7BUFhMB1oYMe6pcVs664dXJETgJXzB5dXNaBY1Uv8TE1ylVVwFDNyj7rkgqpzZxbcYtkLFk4hPKUZ0Wy9JFEeUsCVkJSWxqja2b_461FJb5H-dc8vlU_NGP3ec1snymF8yogfm4VOG8-PDz6Hy9Bdyy7cWDB-BuAE-ZiJz_EEm7BX87QDkZby97moiKol2nHR1oSxXNLTcaFq80SczETcn7pPPBAJa-EuG5QnQ5h2Hdc7R1EYsxilp6OJrbF-Z2tY6p3aZFP3OZGaw2ELraSvxRWWZHUVAZoAW537S0bHoQA2yvKz3yRWGgrkZyLs7a0__23T71qa96lUI79vlCuu0_20GWM3xvxJjSJJeuVjEAqKJwZWS8-_BgypOLnW9g_Cmg__HKiknklbeayAHSOZz2pOlxfqMqmTlyq2MTYpPCQXfFpleF9kTvGNz3tNjBmbd6ftGnfuMVhu6lWROUNVJ3FoYWfqBrlOOhLDa9oegNYOdmDMTIhopMEzzI1Jq9gaPLzw9H4CASdKP7MHYDb73-G6FGDt6_ao_oW3Brqw3SlE2UCA-RkJVHl5D72llXMb82o08oEYCjhO5kMD8DQc8V3cz8c3xIrZYE00DXeYXfZh_VIvis26_OdlyIIdnDW-weqb-uuTM86i1i/download",
        # Body03
        "https://public.boxcloud.com/d/1/b1!XKmHMtUSKIcd9G_yh8A_ZQTBD90Zrx5sjgH2maj9uYj0_jASS08R5ximFaI0xJTvMFWN-la9L25spzpBeIiqu55ErlVuMLoWXuCMGz7mp3C2EomSPJVWN0lrUsI-7V6SntTcXXGjDsvDynozvv7iMQTBHi76Ve3gOMhC62UfPhseseVxx-psS2XKSovxEpESJsfRXt8ssUREUV5za9g6e5NfofGULlKZm62oPq3af-PHx60fo37teovG3Z4UcmkV68tdCTTS1mx0gm-meOXfB1LaVNQqKgvRHUj9JMYzgnWFYNKYsG-fr5nOCGYv19QfczN9VUqdBY7Oh8cTkPAT5QWR-4LvSmcfGZiPdjWHH3b97s4-DNh7tPDLDz-FAKD3IzBV2ui2sGTlz0OmjcRyNc8pXMbr3o02Ury-2lJD6STRypTRGw33H6bNCgtTAiAV7eDXXGxivB1csU_JBZxskcjElwjkleEHKNXZxq0xZdlBZdzk-mhAh-iS0-LA4nGP3-7VYRcdlRVjOVWMmVYHDFoVpU2-AWSMEn9P497SIOEsIF5C0HM-A7F_sagYph9a1DyZ-wfAqjtJoeatK3eVDgDsbktpfb0Xj4s0lzFqfcy9OUqWUw51uifggQadlB9C9u1unw2aEWb5VZQltK3skOA98xRzciVevjvMGb5TX0cXxfONlTCVMuzBSggwDg9z_ZjRExPcUL854RKuTqPDENRyae3ZlIrSLvv1rrCWPMUc-6ccDxFzL5XLf6Z9hxfWWvfr_AwD15E8JSVT_n5624emJssnFUET2BeBvHqmFV3RD_DkoJs1NJQZWsQIf1Rse-5HrBMZ2uo2AgY3UykRp3nOKQTXJzsEUHlItr4qkfanXg7UeCAqQeq1BKfLkMFdXwwxKi2vV5rnq9nG77fhxUhHsDRHXy3C2LStZ9RPgUSnwOvlMsN5hWUDT3zGXijxpHV-g_14VMWcwcPIGJpm8Uc8HlD733NEWGeMWPALawY8VBXB5Pr2VGWkHcwVNItGWXNhrYGlUTz9fj0lesGYfGu7kAa-cJB-WS_LHY7-jro1TrM2UhVT35GtMN3J6KxdbLMX6eE42bgRtD3Oh_aMsYf-PkAz6GnDr1e0KLsaxnLiHUMeR3E9vc5EFrzo66XQx5OzduM_ia_FIr_DbD-cJp_Ct0szxdzLRx07jwcopU0jdG6GS4KV-WzNdYqv2i628XYRuOe70qc4V-EMUJQnCcSH3KBeFUAZJ6FqxsQ5JtUwvCBxwDfC2QnzLeNSkTu0LlZZe1jy1MTtp2vjed8InW-tfMZYVK5SRpvTMvB4Hx3L3DFFptNfMN6vF_a3L3JMsnttCQYrehySdU8eIArwYBCLqYHLHGFRl9V6FG-_Ze7DeqZ9IxvAjZyFqQPhVCQZ8wEpLxg06X1eFWlHJxXQUaJp/download",
        # Body04
        "https://public.boxcloud.com/d/1/b1!Jk-z_s3x8t3UN2qwWrN9DjdCPgAcSXXkkUuGTjj0FeqGM46BuoMhfjnu16Kl6Tn6DXCo5T6a7_bzpJX0llE4pNgpFl-kwkKdZbCmaTXgryvNEopXCVh5N1poZhI47dHmfhsgZpNB_ARW9m4gTrtdHPGFEMKZlVXVqRBuJHk5Iw2S77jsVRgy4s8VpnCzVjGL1ic06WXObzeNtSPDvaRtoUApiEBZw5EiwJgtewxFvAZANdTFYaxwqjd7ksR4QrdIK0xDsXKzgqr6IG_z8IDvcndpxYlxzMg1aohat3vCSPqvgcrkPmvaFwtscX7ZkJTGZts9Kd6K2VyLUVlKaocx5kiJniKUpZpnYhON4m1Mex7b1MrGras5-LkttnwalmRGQfEb2o97vqe9LNqmF3rMoWh-Ab3yImVXES76a-Os9hFugelcNJvub0duMFqDn5aGvTxwviqC1XQmoWRkvwmAS5qIbPb2N7RnjnFFvIbPsooL4c8txRFCIkS6ECGxHUFHSopcgkctLF85PXBHX83fEZVxBFG-API0Vat_UtHw3gX3y5yrkCuPzHgbvV1A0sbm5QwDDz_wdW5MPUf4SOyihQx7vDBH_XHsiwtmQ7RKacmhtT8x1QjQUp4_PJCd263Q7xn2CcIvAuIt6PD7BBqkbPYo7eHk4l2kbMCl3tuRQqzttJ5VqsbJC1LXRJzcWHRfaZ1qg-zv_wdCRaqXGLwuxmUJ11TXbBhzZKEQV2tnDAnsTdHpE4nfuYOmY-lQ_hRi6gXLSDjPReUNuzb8ErXsDRM8FlmyKsRJuP0WUdZnt3RT_ti4Xqg2b5VngqezgK9y11oXmLpZI4Lf39Oes4p97E6E9DKPZGyrRRBwGxHueOndDxxsliDyJ77P2_jszAkEWHPUu-eAKPvcHW7n880mxsKtFVg7NFQ7e8zL6LSgZLBubDTstwvbFV1t__6DT2yC60SwCg_a9oqU9CVMCgf8U5ADOSOq_3duJ6xJQDO0JO7e3PVQ6kCiHoMnRF9kFK0iaghFFl-XwKxwNgTlzC4FhihQGXsBefiJSqfveyIToaBsMfdI1ZihukGomGedcHMCTrYcqoIK34NZyd5zY3EiSr9IFlVBNJQAYUVGETmij6E0u-atw9fblgbiTtCctAn17iitN4ulrSu_AP1p4sBAZBhgQmp4PTysdIFPAHjn2_s8pEkG3ev1FLTZ9iW1uh7IZwyQkrWEuKdUkt8KjTI4w6TolnTFPdpJTXpSSm401qnjNuaptbmdGyzYEyx7FDvBd7XchNRwvUL84Q9avwg1bK_jS4gnKjnRzyD2v61MEb2sXKZIjFJBg7hisfrKEPAWFFdTn4g-5PP2JpGoXPcZiiFL6KhcLvbPoc5jka_O1njvniJSyYq6GwqAR3kbmyKrgJ-0Ji91HYpiIVjez5tM4bRe/download",
    ]
}


def download_file_with_progress(url, output_path):
    """
    Downloads a file from a URL with a stable progress bar.
    """
    print(f"Downloading: {os.path.basename(output_path)}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            leave=True,
            file=sys.stdout,
            dynamic_ncols=True 
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
                
        print(f"Downloaded successfully: {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"\nDownload error {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def extract_tar_gz(tar_path, extract_to):
    """
    Extracts a .tar.gz file to the specified directory.
    """
    try:
        if not tarfile.is_tarfile(tar_path):
            print(f"Error: {tar_path} is not a valid tar file.")
            return False  # Return False on failure

        print(f"Extracting {os.path.basename(tar_path)} to {extract_to}...")

        with tarfile.open(tar_path, "r:gz") as tar_ref:
            members = tar_ref.getmembers()
            for member in tqdm(members, desc="Extracting files", file=sys.stdout, leave=False):
                # Using 'data' filter to avoid deprecation warnings in newer Python versions
                if hasattr(tarfile, 'data_filter'):
                    tar_ref.extract(member, path=extract_to, filter='data')
                else:
                    tar_ref.extract(member, path=extract_to)

        print(f"Extraction successful.")
        return True # Return True on success

    except Exception as e:
        print(f"Unexpected error while extracting {tar_path}: {e}")
        return False

def generate_train_test_lists(data_root, train_ratio=0.8):
    """
    Scans the data directory, shuffles images, and creates train/test lists.
    """
    print(f"\nGenerating train/test lists for ADN (Split ratio: {train_ratio})...")

    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    paths_list = []

    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.lower().endswith(image_extensions):
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, data_root)
                rel_path = rel_path.replace(os.sep, "/")
                paths_list.append(rel_path)

    if not paths_list:
        print("No images found to generate lists.")
        return

    paths_list.sort()
    random.seed(RANDOM_SEED) 
    random.shuffle(paths_list)

    split_index = int(len(paths_list) * train_ratio)
    train_files = paths_list[:split_index]
    test_files = paths_list[split_index:]

    train_txt_path = os.path.join(data_root, "train_list.txt")
    test_txt_path = os.path.join(data_root, "test_list.txt")

    with open(train_txt_path, "w") as f:
        for path in train_files:
            f.write(f"{path}\n")

    with open(test_txt_path, "w") as f:
        for path in test_files:
            f.write(f"{path}\n")

    print(f"Successfully generated lists:")
    print(f" - Train: {len(train_files)} images -> {train_txt_path}")
    print(f" - Test:  {len(test_files)} images -> {test_txt_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Download RPI Medical Dataset (tar.gz) and generate ADN train/test lists."
    )
    
    # Path arguments
    parser.add_argument("-d", "--data_dir", type=str, default="data_rpi", help="Target dir.")
    
    # Selection arguments
    parser.add_argument("--body_part", type=str, choices=["head", "body", "all"], default="all", help="Body part to process.")
    
    parser.add_argument("--num_head", type=int, default=None, help="Number of HEAD files to download (default: all).")
    parser.add_argument("--num_body", type=int, default=None, help="Number of BODY files to download (default: all).")

    # Other settings
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train set ratio (0.0-1.0).")
    parser.add_argument("--skip_download", action="store_true", help="Skip download, only generate lists.")
    
    parser.add_argument("--delete_archive", action="store_true", help="Delete the .tar.gz file after successful extraction to save space.")

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # --- Select URLs based on arguments ---
    selected_urls = []

    # 1. Process HEAD
    if args.body_part in ["head", "all"]:
        head_urls = URLS["head"]
        if args.num_head is not None:
            # Take only the requested number of files
            head_urls = head_urls[:args.num_head]
            print(f"Selected {len(head_urls)} HEAD files (out of {len(URLS['head'])})")
        selected_urls.extend(head_urls)

    # 2. Process BODY
    if args.body_part in ["body", "all"]:
        body_urls = URLS["body"]
        if args.num_body is not None:
            # Take only the requested number of files
            body_urls = body_urls[:args.num_body]
            print(f"Selected {len(body_urls)} BODY files (out of {len(URLS['body'])})")
        selected_urls.extend(body_urls)

    # --- Download Loop ---
    if not args.skip_download:
        if not selected_urls:
            print("No URLs selected. Check your --body_part or --num_* arguments.")
        
        print(f"Total files to download: {len(selected_urls)}")

        for i, url in enumerate(selected_urls):
            filename = url.split("/")[-1]
            if not filename.endswith(".tar.gz"):
                filename = f"archive_{i}.tar.gz"

            tar_path = os.path.join(args.data_dir, filename)

            print(f"\n--- Processing file {i+1}/{len(selected_urls)} ---")

            if os.path.exists(tar_path):
                print(f"Archive {filename} already exists. Skipping download.")
                # We assume if it exists, it might need extraction, 
                # but we generally skip re-downloading.
            else:
                success = download_file_with_progress(url, tar_path)
                if not success:
                    continue

            # Extract and optionally delete
            extraction_success = extract_tar_gz(tar_path, args.data_dir)
            
            if extraction_success and args.delete_archive:
                print(f"Deleting archive to save space: {filename}")
                try:
                    os.remove(tar_path)
                except OSError as e:
                    print(f"Error deleting file: {e}")

    # --- Generate Lists ---
    generate_train_test_lists(args.data_dir, train_ratio=args.train_ratio)

if __name__ == "__main__":
    main()
