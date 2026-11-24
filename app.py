
import numpy as np
import joblib
import os

from flask import Flask, render_template, request

app = Flask(__name__)

# Load machine learning model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "california.joblib")

obj = joblib.load(MODEL_PATH)

model = obj['model']
cols = obj['columns']


# Field information
FIELD_INFO = {
    "MedInc": ("Median Income", "Median income of households"),
    "HouseAge": ("House Age", "Median age of houses"),
    "AveRooms": ("Average Rooms", "Average number of rooms"),
    "AveBedrms": ("Average Bedrooms", "Average bedrooms"),
    "Population": ("Population", "Area population"),
    "AveOccup": ("Average Occupancy", "Average household occupancy"),
    "Latitude": ("Latitude", "Geographical latitude"),
    "Longitude": ("Longitude", "Geographical longitude")
}


@app.route('/', methods=['GET', 'POST'])
def Main():

    prediction = None
    error = None
    submitted_values = {}

    if request.method == 'POST':

        In = []

        try:
            for i in cols:

                value = request.form.get(str(i), '').strip()

                if value == '':
                    raise ValueError(f'Please enter {i}')

                v = float(value)

                if not np.isfinite(v):
                    raise ValueError(f'Invalid value for {i}')

                In.append(v)
                submitted_values[str(i)] = value

            Input = np.array([In])

            out = model.predict(Input)

            prediction = float(out[0])

        except ValueError as e:
            error = str(e)

        except Exception as e:
            app.logger.exception("Prediction error")
            error = "Something went wrong while predicting. Please check your inputs."

    fields = []

    for i in cols:

        label, hint = FIELD_INFO.get(
            str(i),
            (str(i).replace('_', ' ').title(), 'Enter a numeric value')
        )

        fields.append({
            'key': str(i),
            'label': label,
            'hint': hint
        })

    return render_template(
        'index.html',
        fields=fields,
        prediction=prediction,
        error=error,
        submitted_values=submitted_values
    )


if __name__ == '__main__':
    app.run(debug=True)
