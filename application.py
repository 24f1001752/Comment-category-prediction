from flask import Flask, request, render_template
import pandas as pd

from mlproject2_src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    try:
        # Required
        comment = request.form.get("comment", "")

        # Optional numeric/flags (default to 0 if missing/empty)
        def to_int(name, default=0):
            val = request.form.get(name, "")
            if val is None or str(val).strip() == "":
                return default
            return int(float(val))  # handles "1", "1.0"

        data = CustomData(
            comment=comment,
            upvote=to_int("upvote"),
            downvote=to_int("downvote"),
            emoticon_1=to_int("emoticon_1"),
            emoticon_2=to_int("emoticon_2"),
            emoticon_3=to_int("emoticon_3"),
            if_1=to_int("if_1"),
            if_2=to_int("if_2"),
            race=to_int("race"),
            religion=to_int("religion"),
            gender=to_int("gender"),
            disability=to_int("disability"),
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df, return_label_names=True)

        # pred is an array-like (single row)
        return render_template("home.html", results=str(pred[0]))

    except Exception as e:
        # show error on page instead of crashing
        return render_template("home.html", results=f"Error: {e}")
    

if __name__ == "__main__":
    # Run on localhost
    app.run(host="0.0.0.0", port=5000, debug=True)
