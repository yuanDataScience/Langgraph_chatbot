import os

import aiofiles
from aiofiles.os import makedirs
from fastapi import UploadFile

DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50  # 50 megabytes


async def save_file(file: UploadFile) -> str:
    """
    accept a UploadFile from FastAPI and save the file in specified location
    :param file: UploadFile specified by users from fastAPI interface
    :return: path of the saved file
    """
    await makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    async with aiofiles.open(filepath, "wb") as f:
        while chunk := await file.read(DEFAULT_CHUNK_SIZE):
            await f.write(chunk)
    return filepath
