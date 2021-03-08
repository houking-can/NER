import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq, name):
    length = len(char_seq)
    res = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-'+name:
            if 'pre' in locals().keys():
                res.append(pre)
                del pre
            pre = char
            if i + 1 == length:
                res.append(pre)
        if tag == 'I-'+name:
            if 'per' in locals().keys():
                pre += char
            else:
                pre = char
            if i + 1 == length:
                res.append(pre)

        if tag not in ['I-'+name, 'B-'+name]:
            if 'per' in locals().keys():
                res.append(pre)
                del pre
            continue
    return res

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
