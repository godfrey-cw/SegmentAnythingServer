import requests
from PIL import Image
from io import BytesIO
import pickle as pkl

with open("auth/ipaddress.txt", "r") as f:
    SAMSERVEIP = list(f.readlines())[-1].split("\n")[0]

PREDICTIONS_API = "https://" + SAMSERVEIP + ":8443/predictions"
certfile = "auth/mycert.pem"


def auto_maskgen(image: Image, model_scale: str = "b"):
    imbytes = BytesIO()
    image.save(imbytes, format="PNG")
    imbytes.seek(0)
    res = requests.post(
        url=(PREDICTIONS_API + f"/sam_{model_scale}_auto_maskgen"),
        files={"image": imbytes},
        verify=False,
    )
    return res


def predict(
    image: Image,
    prompt_points: list[list[int]],
    prompt_labels: list[int],
    model_scale: str = "b",
):
    imbytes = BytesIO()
    image.save(imbytes, format="PNG")
    imbytes.seek(0)
    ppbytes = BytesIO()
    ppbytes.write(pkl.dumps(prompt_points))
    ppbytes.seek(0)
    plbytes = BytesIO()
    plbytes.write(pkl.dumps(prompt_labels))
    plbytes.seek(0) 
    url = PREDICTIONS_API + f"/sam_{model_scale}_predict"
    print(url)
    res = requests.post(
        url=url,
        files={
            "image": imbytes,
            "prompt_points": prompt_points,
            "prompt_labels": prompt_labels,
        },
        verify=False,
    )
    return res
