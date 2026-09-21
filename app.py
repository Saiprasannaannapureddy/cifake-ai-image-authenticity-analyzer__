from pathlib import Path
import uuid

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from predict import predict_image

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", error="Please select an image.")

    file = request.files["image"]

    if not file or file.filename == "":
        return render_template("index.html", error="Please select an image.")

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Supported formats: JPG, JPEG, PNG and WEBP."
        )

    # Use a generated name so uploaded filenames cannot overwrite each other.
    safe_name = secure_filename(file.filename)
    extension = Path(safe_name).suffix.lower()
    filename = f"{uuid.uuid4().hex}{extension}"
    filepath = UPLOAD_FOLDER / filename
    file.save(filepath)

    try:
        result, confidence = predict_image(filepath)
    except Exception as exc:
        filepath.unlink(missing_ok=True)
        return render_template(
            "index.html",
            error=f"Could not analyze the image: {exc}"
        )

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        image_path=f"uploads/{filename}",
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        error="The image is too large. Please upload an image below 5 MB."
    ), 413


if __name__ == "__main__":
    app.run(debug=True)
