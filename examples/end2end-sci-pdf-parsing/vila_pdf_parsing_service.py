import tempfile
import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import requests
import aiofiles
import aiohttp

from main import build_predictors, pipeline

pdf_extractor, vision_model1, vision_model2, pdf_predictor = build_predictors()
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/parse/")
async def detect_upload_file(
    pdf_file: UploadFile = File(...), relative_coordinates: bool = False
):

    with tempfile.TemporaryDirectory() as tempdir:
        async with aiofiles.open(f"{tempdir}/tmp.pdf", "wb") as out_file:
            content = await pdf_file.read()  # async read
            await out_file.write(content)  # async write

        layout_csv = pipeline(
            input_pdf=Path(f"{tempdir}/tmp.pdf"),
            return_csv=True,
            pdf_extractor=pdf_extractor,
            vision_model1=vision_model1,
            vision_model2=vision_model2,
            pdf_predictor=pdf_predictor,
            relative_coordinates=relative_coordinates,
        )

    return StreamingResponse(
        io.StringIO(layout_csv.to_csv(index=False)), media_type="text/csv"
    )


@app.get("/parse/")
async def detect_url(pdf_url: str, relative_coordinates: bool = False):

    # Refer to https://stackoverflow.com/questions/35388332/how-to-download-images-with-aiohttp
    with tempfile.TemporaryDirectory() as tempdir:
        async with aiohttp.ClientSession() as session:
            async with session.get(pdf_url) as resp:
                if resp.status == 200:
                    f = await aiofiles.open(f"{tempdir}/tmp.pdf", mode="wb")
                    await f.write(await resp.read())
                    await f.close()

        layout_csv = pipeline(
            input_pdf=Path(f"{tempdir}/tmp.pdf"),
            return_csv=True,
            pdf_extractor=pdf_extractor,
            vision_model1=vision_model1,
            vision_model2=vision_model2,
            pdf_predictor=pdf_predictor,
            relative_coordinates=relative_coordinates,
        )

    return StreamingResponse(
        io.StringIO(layout_csv.to_csv(index=False)), media_type="text/csv"
    )
