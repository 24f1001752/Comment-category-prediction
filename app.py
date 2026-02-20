from flask import Flask, request, render_template
import pandas as pd
import os  
from mlproject2_src.utils import load_object
from mlproject2_src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    results = None
    
    if request.method == "GET":
        return render_template("home.html", results=results)
    
    # POST - Process prediction
    try:
        # Form data
        comment = request.form.get("comment", "")
        def to_int(name, default=0):
            val = request.form.get(name, "")
            return default if val is None or str(val).strip() == "" else int(float(val))

        data = CustomData(
            comment=comment, upvote=to_int("upvote"), downvote=to_int("downvote"),
            emoticon_1=to_int("emoticon_1"), emoticon_2=to_int("emoticon_2"), 
            emoticon_3=to_int("emoticon_3"), if_1=to_int("if_1"), if_2=to_int("if_2"),
            race=to_int("race"), religion=to_int("religion"), gender=to_int("gender"),
            disability=to_int("disability")
        )
        
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        
        # Get numeric prediction first
        pred = predict_pipeline.predict(pred_df, return_label_names=False)
        pred_num = int(pred[0])
        
        # Get category names
        category_names = ["CLEAN", "TOXIC", "SEVERE TOXIC", "THREAT"]
        try:
            le = load_object(os.path.join("artifacts", "label_encoder.pkl"))
            category_names = [str(c) for c in le.classes_]
        except:
            pass
            
        category_names = ["clean", "toxic", "severe toxic", "threat"]
        pred_meaning = category_names[pred_num]
        results = f"{pred_num} ({pred_meaning}): {comment}"

        
    except Exception as e:
        results = f"Error: {str(e)}"
    
    return render_template("home.html", results=results)


if __name__ == "__main__":
    # Run on localhost
    app.run(host="0.0.0.0", port=5000, debug=True)
