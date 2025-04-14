from detection import ham, lof, zscore, expma, mahalanobis, multi_ham, dbscan, wlf
from identification import value_counting, subsetting
from configs.run_configs import RunConfig


def match_detection_tech(tech): 
    match tech: 
        case "ham": return ham
        case "lof": return lof
        case "zscore": return zscore
        case "expma": return expma
        case "maha": return mahalanobis
        case "multi_ham": return multi_ham
        case "dbscan": return dbscan
        case "wlf": return wlf
        # case "autoencoder": return autoencoder
        case _: raise AttributeError(f">match_detection_tech({tech}) is not available")
        

def match_identification_tech(tech): 
    match tech: 
        case "value_counting": return value_counting
        case "subsetting": return subsetting
        case _: raise AttributeError(f">match_identification_tech({tech}) is not available")


def calculate_metrics(configs : RunConfig, id_exp = None): 
    # for tech in configs.detection.techniques: 
    #    match_tech(tech).compute_accuracy(configs)
    
    # voting_aggregation.compute_accuracy(configs, OUTLIER_FP_RATIO=0.1)
    # voting_aggregation.compute_accuracy(configs, OUTLIER_FP_RATIO=0.2)
    # voting_aggregation.compute_accuracy(configs, OUTLIER_FP_RATIO=0.5)
    # voting_aggregation.compute_accuracy(configs, OUTLIER_FP_RATIO=1)
    # voting_aggregation.compute_accuracy(configs, OUTLIER_FP_RATIO=2)
    # voting_aggregation.compute_accuracy(configs, OUTLIER_FP_RATIO=3)

    for tech in configs.detection.techniques: 
        match_detection_tech(tech).compute_bruteforce_accuracy(configs, create_chart=True, id_exp=id_exp)

    for tech in configs.identification.techniques: 
        match_identification_tech(tech).get_common_outliers(configs, id_exp=id_exp)

    # configs.identification.keep_epochs = [0,1]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [2,3]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [4,5]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [6,7]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [8,9]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [10, 11]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [12,13]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [14,15]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [16,17]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [18,19]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)


    # configs.identification.keep_epochs = [0,1,2]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [2,3,4]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [4,5,6]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [6,7,8]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [7,8,9]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [10,11,12]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [12,13,14]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [14,15,16]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [16,17,18]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [17,18,19]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)

    # configs.identification.keep_epochs = [0,1,2,3,4]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [5,6,7,8,9]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [10,11,12,13,14]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [15,16,17,18,19]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)

    # configs.identification.keep_epochs = [0,1,2,3,4,5,6,7,8,9]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [10,11,12,13,14,15,16,17,18,19]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)

    # configs.identification.keep_epochs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)
    # configs.identification.keep_epochs = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    # print(f"configs.identification.keep_epochs = {configs.identification.keep_epochs}")
    # value_counting.get_common_outliers(configs, id_exp=id_exp)