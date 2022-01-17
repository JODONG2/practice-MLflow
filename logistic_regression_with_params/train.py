import sys
import argparse
import mlflow
import mlflow.sklearn

import numpy as np
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    mlflow.sklearn.autolog()
    X= np.random.randint(5,size=(80,)).reshape(-1,4)
    y= np.random.randint(2,size=(20,))

    LR=LogisticRegression(
        solver= sys.argv[1], penalty=sys.argv[2], l1_ratio=float(sys.argv[3])
        )
    with mlflow.start_run() as run:
        LR.fit(X,y)

    score=LR.score(X,y)
    print(f"Score: {score}")
