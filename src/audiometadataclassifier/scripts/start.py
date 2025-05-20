import uvicorn

def run():
    uvicorn.run("src.audiometadataclassifier.main:app", reload=True)

    