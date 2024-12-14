1. In model/GPUsetting
```
bash cuda_install.sh
sudo reboot
```

2.
```
bash setup_virtualenv.sh
```

3. In model
```
pyenv activate [your virtualenv name]

bash model_server_setting.sh
```

4.
```
bash main.sh
# model cache would be located in sql-helper/.cache
```