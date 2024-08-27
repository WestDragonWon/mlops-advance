import configparser

credentials_file = '~/.aws/credentials'

config = configparser.ConfigParser()
config.read(credentials_file)