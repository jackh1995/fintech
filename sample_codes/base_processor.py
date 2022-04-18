import abc

import pandas as pd


class FeaturesProcessor(abc.ABC):
    """
    Process features and label, classify job title
    """
    @abc.abstractmethod
    def get_features(self, data):
        """
        build features from has been processed data
        use classify_job_title function process job_title feature
        input沒有限制，看是要call一個function去讀資料，還是讀完資料丟進來get_features去做
        Returns: Dataframe
        """
        pass

    @abc.abstractmethod
    def get_y(self, data):
        """
        process y data(first_quota, now_quota)
        Returns: Dataframe
        """
        pass

    @abc.abstractmethod
    def classify_job_title(self, df: pd.DataFrame):
        """
        process job_title, example: if job_title is 總經理 then 1 else 3 etc...
        Returns: Dataframe
        """
        pass


class PredictModel(abc.ABC):
    # TODO: inpurt variable討論
    @abc.abstractmethod
    def predict(self, data, model_file_path, **kwargs):
        """
        use load_model function get model object, input data, return predict result finally
        Returns: Dataframe or dict, result must include ssn, predict_quota
        """
        pass

    @abc.abstractmethod
    def load_model(self, model_file_path):
        """
        load model
        Returns: model object
        """
        pass
