This document provides the details of CLI developed for using the Ocean Protocol platform for
publishing and consuming data and algorithm. it constitutes -

1. Data publishing and retrieval - A publisher publishes data and provides a token to the recipient
   for retrieving the data.

2. Compute to Data (C2D) - Wherein a publisher publishes a dataset, and allows an algorithm to be 
   executed against the data, by minting appropriate tokens to the recipient. 
   Point to note is once an algorithm is published and it's DID is stored, the same algorithm can
   be executed on a separate dataset. This is detailed on teh section Gradio App.

NOTE :: Currently all the CLIs are hardcoded to use "polygon-test" which is the Polygon Testnet 
        Mumbai. It can be changed to a parameter in the next version. Also, a separate document
        for Setup will be added detailing how to setup and run on separate instances.


Data publishing and Retrieval 

Publish

CLI - publishDatasetArgs.py <recipient address> <dataset URL>

Example -  
python3 publishDatasetArgs.py "0x427101Aee61E77dc22386f0ae944d687FE062b60" "https://raw.githubusercontent.com/YodaGB/OceanC2D/main/weather_test.py"

Output -
Just published asset:
  data_nft: symbol=<ContractCall 'symbol()'>, address=0x36B80037215C32eBa247A72bC9bAFb33e3204B9e
  datatoken: symbol=<ContractCall 'symbol()'>, address=0xDe7DCF1476b6C1fa4A654B9C0470860F2186DBB0
  did=did:op:694f01a172d68e9ca561142d72519dfa02c6af06f2559e88af0b7991877c3906

Retrieve
 
LI - GetDataArgs.py <published dataset DID>

Example - retrieving the above published dataset
python3 GetDataArgs.py "did:op:694f01a172d68e9ca561142d72519dfa02c6af06f2559e88af0b7991877c3906"
Output -
 ./datafile.did:op:694f01a172d68e9ca561142d72519dfa02c6af06f2559e88af0b7991877c3906,0/file0
NOTE:: The file name is returned as data DID with filename as file0 as above.
cat ./datafile.did:op:694f01a172d68e9ca561142d72519dfa02c6af06f2559e88af0b7991877c3906,0/file0

Compute To Data (C2D) Flow

Publishing data and algorithm, allowing algorithm to execute on dataset, and issuing 
algorithm and data tokens to the recipent for later execution to consume the result.
CLI - Ocean_Publish_Data_Algo_Params_v0.py
Parameters - 
<recipientAddress> - consumer's address
<dataset_URL> - URL of the dataset currently tested with github raw content
<dataset_name> - Name of the dataset
<algo_URL> - URL of the algorithm currently tested with github raw content
<algo_name>= Name of the algorithm
<p_image> - Docker Image where prerequisites are stored.e.g. requirements.txt for python or package.json for node. 
<p_tag> - Tag in the docker image where the data is retrieved from
<p_checksum> - Digest of the above docker image when pushed.

NOTE:: AS Ocean Protocol docker images are limited,you can create your own image as shown in the example below.

example - To generate base statistics of a CSV dataset.
python3 /home/ubuntu/Ocean_Publish_Data_Algo_Params_v0.py "0x427101Aee61E77dc22386f0ae944d687FE062b60" "https://raw.githubusercontent.com/YodaGB/OceanC2D/main/covid19_cases.csv" "Covid" "https://raw.githubusercontent.com/YodaGB/OceanC2D/main/csv_stats.py" "Generate CSV Stats" "gbzeroai/algo-dockers" "python-panda" "sha256:d9a1bfe72a44542c2dd39ba19f764a54dc7b98125c198628524f9d893e82d3a2"

Output - 
DATA_ddo did = 'did:op:7206e6d0fbb39d4ce92b7ed5d903e58930dae6be46275af6c43e959ffc7524a4'
Just published asset:
  data_nft: symbol=<ContractCall 'symbol()'>, address=0xaAb1E1f60FE6D0eb71f01aCa86c69d9FF59cF839
  datatoken: symbol=<ContractCall 'symbol()'>, address=0xD5D401a43f3C6D73aB32E24Db14bE2Cf89EE7899
DATA_ddo did = 'did:op:7206e6d0fbb39d4ce92b7ed5d903e58930dae6be46275af6c43e959ffc7524a4'
ALGO_data_nft address = '0xa711A0B99FE49F3fe4ae2AB8f67dd30e86C64Be0'
ALGO_datatoken address = '0xae7c6fC64aCfB74A4c1C344196f95fE4aDFe458C'
ALGO_ddo did = 'did:op:b13a565b66d05821478d4012ca40f0151c2de71b160a3a55dbcad649a7408214'

NOTE :: Most important are the DATA_ddo_did and ALGO_ddo_did for next step of execution. The consumer
        uses this as shown below.

Execution - Starting an execution job by the consumer.
CLI - ExecuteCompute_v0.py
Parameters
<DATA DID> - Published Data DID in step 1 above.
<ALGO DID> - Published Data DID in step 1 above.

python3 /home/ubuntu/ExecuteCompute_v0.py "did:op:7206e6d0fbb39d4ce92b7ed5d903e58930dae6be46275af6c43e959ffc7524a4" "did:op:b13a565b66d05821478d4012ca40f0151c2de71b160a3a55dbcad649a7408214"
Output -
Started compute job with id: cd09c456858e423c944eee2252585df4

NOTE :: The compute job is required for monitoring progress and getting result.

Monitoring - Monitoring the job and getting execution result.
CLI - Monitor_job_v0.py

Parameters -
<DATA DID> - Data DID used in execute.
<Job ID> - returned as Execute output.
<Dump File> - A file in which the execution output is dumped.

python3 /home/ubuntu/Monitor_job_v0.py "did:op:7206e6d0fbb39d4ce92b7ed5d903e58930dae6be46275af6c43e959ffc7524a4" "cd09c456858e423c944eee2252585df4" "dump_29032023_0"

NOTE :: The output is dependent on the algorithm. It can be just a text file in case of the above example.
        It can also be a pickled file containing a model or a numpy array.based on teh same, it has to be
        processed. see gradio example notes.

IMPORTANT TO SAVE THE DID :: ALGO_ddo did = 'did:op:b13a565b66d05821478d4012ca40f0151c2de71b160a3a55dbcad649a7408214'

Publishing a data set and allowing it to execute against an already published algorithm.

CLI - publishdatasetArgs_AllowAlgo_v0.py

Parameters
<recipient address> - the recipient whose dataset is allowed to execute on already published algorithm.
<Dataset URI> - The dataset on which the algorithm is to be executed.
<Dataset name > - Name of the dataset.
<Algorithm DID> - DID of already published algorithm.

This is useful in case like Gradio App.

NOTES ON CLIs :: There is whole lot of console logging on ocean lib calls, and it is painful to scroll up 
                 especially when publishing an algorithm to retrieve the algorithm and dataset DIDs. 
                 It has been tried to run using stdout redirection but for some reason Ocean lib crashes.
                 In next version it can be tried writing the relevant DIDs in a file which can be later accessd.

Gradio - A small application is created for executing algorithms against input which can be for prediction to
         an already existing model from previously published algorithm, or an algorithm which executes on the 
         dataset.
         The code for the same is in Gradio_example_v0.py.
         How processing is done depends on algorithm choice - 
         In case of Linear Regression we do not do any execution just unpickle an already trained model and use the
         predict method.
         For dataset Statistics, we use functions publishing dataset for an algo, execute compute and monitoring (same as CLI) and then i        	 interpret the output which is plain text..
         The code will require modification as newer algorithms are added.

Writing algorithms for Ocean

algorithms for Ocean is written in a way, to accept input and produce outputs. 2 samples of the same are
available, linear_regression.py and CSV_stats.py. Linear regression is trained in the Housing dataset.
CSV_stats is initiall tested on Covid dataset and then subsequently tested using the Weather dataset.
        .
           




