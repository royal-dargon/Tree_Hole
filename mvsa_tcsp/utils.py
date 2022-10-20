# 这个是对模型训练的配置文件
import argparse


def get_argparse():
    parser = argparse.ArgumentParser()

    general_group = parser.add_argument_group(title="general")
    general_group.add_argument("--dataset", type=str, default='single')
    general_group.add_argument("--do_trans", action="store_true", default=False)


config_single = {
    "w2i_model": dict(),
    "translation": dict()
}

# 目前只是对single的配置
config_single["w2i_model"]["source_size"] = 768
config_single["w2i_model"]["encoder_hidden_size"] = 100
config_single["w2i_model"]["encoder_num_layers"] = 1
config_single["w2i_model"]["encoder_dropout"] = 0
config_single["w2i_model"]["decoder_input_size"] = 200
config_single["w2i_model"]["decoder_hidden_size"] = 100
config_single["w2i_model"]["decoder_num_layers"] = 1
config_single["w2i_model"]["decoder_dropout"] = 0
config_single["w2i_model"]["target_size"] = 20

config_single["translation"]["lr"] = 1e-3
