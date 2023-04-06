This repo contains all the files which are used in Huggingface Spaces for 'Zero2AI Auto Execute' App.
The main app.py uses Infura fateway to connect to the Polygon Testnet blockchain.
There is a file in main Ocean gradio_For_Local_Test_key.py which is the exact replica except, in app.py the first line in the application sets the environment variable for Infura gateway access which is the Infura key.
The publisher and consumer accounts and their corresponding private keys are hardcoded in the app, which is fine as we are executing it in the testnet. If we need to go to rpoduction we need to think about security also for point 3.
the app shows functionalities for 
- Using previously published models for prediction(Linear Regression) and Evaluation (Random forest).
- Using an original algorithm to run against a new dataset by allowing the already published original algorithm to execute against the new dataset (CSV Stats).
- We are not currently allowinhg publishing algorithm using the app, which is technically feasible.
- Wherever passing the dataset as an input please use the raw git as others doesn't work.
We will try to improve the app, by adding additional algorithms and Output types e.g. AUC/ROC curves as gradio output.
