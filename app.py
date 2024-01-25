import os
import uuid
import time
from enum import Enum
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, Path, Query, Response
from fastapi.middleware.cors import CORSMiddleware

import torch

from model import SDModel
from arguments import api_config, output_dir

pipeline = SDModel()

app = FastAPI(
    title="View layer service API for UI",
    version='0.1.1'
)

origins = [
    f"http://{api_config.host}",
    f"http://{api_config.url}",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Check health of application
@app.get('/notify/v1/health')
def get_health():
    """
    Usage on K8S
    readinessProbe:
        httpGet:   path: /notify/v1/health
            port: 80
    livenessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    :return:
        dict(msg='OK')
    """
    return dict(msg='OK')

class BoolJudgement(str, Enum):
    yes = 'yes'
    no = 'no'

class Text2ImgPayload(BaseModel):
    key: str = Field("", description='Your API Key used for request authorization.')
    prompt: str = Field(..., description="Text prompt with description of the things \
                                         you want in the image to be generated.")
    negative_prompt: str = Field("", description="Items you don't want in the image.")
    width: int = Field(512, ge=128, le=1024, description='Max Height: Width: 1024x1024.')
    height: int = Field(512, ge=128, le=1024, description='Max Height: Width: 1024x1024.')
    samples: int = Field(1, ge=1, le=4, description='Number of images to be returned in response. \
                                                        The maximum value is 4.')
    num_inference_steps: int = Field(20, gt=0, lt=52, description='Number of denoising steps.')
    safety_checker: BoolJudgement = Field(BoolJudgement.yes, description='A checker for NSFW images. \
                                               If such an image is detected, it will be replaced by a blank image.')
    enhance_prompt:	BoolJudgement = Field(BoolJudgement.yes, description='Enhance prompts for better results; \
                                                                          default: yes, options: yes/no.')
    seed: int = Field(-1, description='Seed is used to reproduce results, same seed will give you same image in return again. \
                                         Pass null for a random number.')
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description='Scale for classifier-free guidance \
                                                                    (minimum: 1; maximum: 20).')
    multi_lingual: BoolJudgement = Field(BoolJudgement.yes, description='Allow multi lingual prompt to generate images. \
                                                                        Use "no" for the default English.')
    panorama: BoolJudgement = Field(BoolJudgement.yes, description='Set this parameter to "yes" to generate a panorama image.')
    self_attention: BoolJudgement = Field(BoolJudgement.no, description='If you want a high quality image, set this parameter to "yes". \
                                                                          In this case the image generation will take more time.')
    upscale: BoolJudgement = Field(BoolJudgement.no, description='Set this parameter to "yes" if you want to upscale the given image resolution two times (2x). \
                                                                  If the requested resolution is 512 x 512 px, the generated image will be 1024 x 1024 px.')
    embeddings_model: str = Field('', description='This is used to pass an embeddings model (embeddings_model_id).')
    webhook: str = Field('', description='Set an URL to get a POST API call once the image generation is complete.')
    track_id: str = Field('', description='This ID is returned in the response to the webhook API call. \
                                           This will be used to identify the webhook request.')


@app.post('/api/v1/text2img')
async def generate_img_from_text(args: Text2ImgPayload):

    generator = torch.Generator("cuda").manual_seed(args.seed) if args.seed != -1 else None

    prompt = "nsfw, full body, blonde hair, pink areola"
    output = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.samples,
        generator=generator,
        eta=0.5,
        safety_checker=args.safety_checker==BoolJudgement.yes
    )

    output_list = []
    for i, sample in enumerate(output.images):
        
        output_origin_name = f'{args.key}{prompt}{args.seed}{time.time()}{i}'
        new_name = uuid.uuid3(uuid.NAMESPACE_URL, output_origin_name)
        output_path = f"{output_dir}/{new_name}.png"
        sample.save(output_path)
        output_list.append(f'http://{api_config.access_url}/img/{new_name}.png')
    
    return {
        "status": "success",
        "generationTime": 1.0,
        "id": 12347568,
        "output": output_list,
        "meta": {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "H": args.height,
            "W": args.width,
            "guidance_scale": args.guidance_scale,
            "n_samples": args.samples,
            "steps": args.num_inference_steps,
        }
    }


@app.get("/img/{img_name}")
def get_img(
    img_name: str = Path(..., description="The path of img, with or without filename extension")):
    img_path = f"{output_dir}/{img_name}"
    if os.path.exists(img_path):
        with open(img_path, "rb") as image_file:
            image_data = image_file.read()
        return Response(content=image_data, media_type="image/png")
    else:
        return {'data': ''}


def main():
    from arguments import api_config
    import uvicorn
    uvicorn.run('app:app',
                host=api_config.host,
                port=api_config.port,
                reload=True,
                reload_dirs=['.'])
    
if __name__ == '__main__':
    from arguments import api_config
    import uvicorn
    uvicorn.run(app, host=api_config.host, port=api_config.port)