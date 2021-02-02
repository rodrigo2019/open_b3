#### Usage:
* Execute `pip install -r requirements`
* Copy and rename `.env_conf_template.json` to `.env_conf.json`.
* Edit `.env_conf.json` using your credentials. (If you don't have any credentials, contact your broker to hire the 
  MetaTrader 5 platform)
* Execute:
  
        cd ./tools
        python get_data_mt5.py
        python generate_dataset.py
        python preprocess_dataset.py
* At this point you should have your own dataset ready for training, then execute:

        cd ../training
        python train.py


#### Limitations:
* As MetaTrader5 works based on Windows platform, this repository in only compatible with Windows.