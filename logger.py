import logging
import os

log_dir = '.experiment/log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 创建一个logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
log_path = log_dir + "/"  # 指定文件输出路径

logname = log_path + 'out.log'  # 指定输出的日志文件名
fh = logging.FileHandler(logname, encoding='utf-8')  # 指定utf-8格式编码
fh.setLevel(logging.DEBUG)

# 创建一个handler，用于将日志输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

if __name__ == '__main__':
    logger.debug("User %s is loging" % 'admin')
