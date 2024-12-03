import uvicorn
import pickle
import pandas as pd
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from typing import List

app = FastAPI()

with open("models.pkl", "rb") as f:
    targetenc = pickle.load(f)
    oheenc = pickle.load(f)
    ridgemodel = pickle.load(f)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def preprocessing(df):
    # cleaning item data
    df = df.drop(['torque', 'selling_price'], axis=1)
    df['mileage'] = pd.to_numeric(
        df['mileage'].str.replace(" kmpl", "").str.replace(" km/kg", ""), errors='coerce'
    ).astype(float)
    df['engine'] = pd.to_numeric(
        df['engine'].str.replace(" CC", ""), errors='coerce'
    ).astype(float)
    df['max_power'] = pd.to_numeric(
        df['max_power'].str.replace(" bhp", ""), errors='coerce'
    ).astype(float)
    df[['mileage', 'engine', 'max_power', 'seats']] = df[
        ['mileage', 'engine', 'max_power', 'seats']].fillna(0)
    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)

    # target encoding 'name' variable
    df['name'] = targetenc.transform(df[['name']])

    # One Hot Encoding all categorical variables
    df['seats'] = df['seats'].astype('object')
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    one_hot_encoded = oheenc.transform(df[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded,
                              columns=oheenc.get_feature_names_out(categorical_columns))
    df_encoded = pd.concat([df, one_hot_df], axis=1)
    df = df_encoded.drop(categorical_columns, axis=1)

    return df

@app.get('/')
def index():
    return {'message': 'Car price prediction ML API'}


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.model_dump()])
    df = preprocessing(df)
    prediction = ridgemodel.predict(df)

    return prediction


@app.post("/predict_items")
def predict_items(items: UploadFile = File(...)) -> StreamingResponse:
    # reading file to pandas
    try:
        df = pd.read_csv(items.file)
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        items.file.close()

    # Items validation for every object in df
    try:
        Items(objects=df.to_dict(orient="records"))
    except ValidationError as exc:
        print(repr(exc.errors()[0]['type']))

    # cleaning data and predicting price
    df_output = df.copy() # need it to return unprocessed data
    df = preprocessing(df)
    prediction = ridgemodel.predict(df)
    df_output['selling_price'] = prediction

    # output to csv
    stream = io.StringIO()
    df_output.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"

    return response


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)