import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    mlflow.sklearn.autolog()
    X= np.random.randint(5,size=(80,)).reshape(-1,4)
    y= np.random.randint(2,size=(20,))

    penalty = 'elasticnet'
    l1_ratio= 0.1

    LR=LogisticRegression(solver='saga', penalty=penalty, l1_ratio=l1_ratio)
    with mlflow.start_run() as run:
        LR.fit(X,y)

    score=LR.score(X,y)
    print(f"Score: {score}")
