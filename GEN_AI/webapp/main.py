import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .inference import FinerInference

app = FastAPI()
inference = None


@app.on_event("startup")
def startup_event():
    global inference
    inference = FinerInference()


@app.get("/", response_class=HTMLResponse)
def index():
    return (
        """
        <html><head><title>FiNER Web</title></head>
        <body>
        <h2>Financial NER (FiNER-139)</h2>
        <form method="post" action="/predict">
            <textarea name="text" rows="8" cols="80"></textarea><br/>
            <button type="submit">Extract Entities</button>
        </form>
        </body></html>
        """
    )


@app.post("/predict")
async def predict(request: Request):
    form = await request.form()
    text = form.get("text", "")
    result = inference.predict(text)
    return JSONResponse({"entities": result})


if __name__ == "__main__":
    uvicorn.run("webapp.main:app", host="0.0.0.0", port=8000)

