from base_processor import PredictModel, FeaturesProcessor
import pandas as pd
import pickle


def get_data(data_path):
    df = pd.read_csv(data_path)
    return df


class QuotaPredict(PredictModel):
    def predict(self, data, model_file_path, **kwargs):
        """
        load model and input data, return predict result finally
        Returns: Dataframe
        """
        model = self.load_model(model_file_path=model_file_path)
        df = self.transform_data(data=data)
        # Read Model object and input data, get reault
        # y_pred = model.predict(df)
        # y_prob = model.predict_proba(df)
        result_example = {'ssn': ['1234F', '2234C', '4546D'], 'predict_quota': [100000, 200000, 300000]}
        predict_result = pd.DataFrame.from_dict(result_example)
        return predict_result

    def load_model(self, model_file_path):
        model = pickle.load(open(model_file_path, 'rb'))
        return model

    def transform_data(self, data):
        data_processor = IndustryProcess()
        df = data_processor.get_features(data=data)
        return df


class IndustryProcess(FeaturesProcessor):
    def get_features(self, data):
        data_dict = data.dict()
        df = pd.DataFrame([data_dict])
        df = self.classify_job_title(df=df)
        return df

    def get_y(self, data):
        data_dict = data.dict()
        df = pd.DataFrame([data_dict])
        df = df.loc[:,['ssn', 'quota']]
        return df

    def classify_job_title(self, df: pd.DataFrame):
        df['job_title'] = df['job_title'].apply(lambda x: '1' if (x[-3:] == '總經理')
                                                              else '3')
        return df