import utils

res = utils.Data('./history_res/20241122172838_1584c654333040b98d6c2897fc3c7950').load_all()
res[1]['materials'][0].__dict__