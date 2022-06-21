from tsa import Quantification, Utils
from tsa import *
from tsa.sniffer import Packet
import glob
import pandas as pd
import os
import sys
import copy
import time

#List all the .tss files in ICSE2020_benchmark_traces
#Run all options (All combinations of Classifiers and Feature reduction), list them in a csv file.

#				rf-classifier      - Classifying information using Random Forest Classifier.
#				knn-classifier     - Classifying information using K-Nearest Neighbors Classifier.
#				nb-classifier      - Classifying information using Gaussian Naive Bayes.
#				fcnn-classifier    - Classifying information using Fully Connected Neural Networks.




#tssfiles = []
#for f in glob.glob("./ICSE2020_benchmark_traces/*1.tss"):
#    tssfiles.append(f)
#
#tssfiles.sort()
#
##print(tssfiles)
#
#for tssfilename in tssfiles:
#    traces1 = Utils.parsetssfiles(tssfilename)
#    traces2 = Utils.parsetssfiles(tssfilename[:-5] + '2.tss')
#    print(tssfilename,                len(traces1))
#    if 'start' in traces1[0][0].load:
#        traces1 = traces1[1:]
#    print(tssfilename[:-5] + '2.tss', len(traces2))
#
#    #Removing the start traces at the beginning.
#    if 'start' in traces2[0][0].load:
#        traces2 = traces2[1:]
#
#    traces = traces1 + traces2
#    print("All added", len(traces))
#
#    for clf_method in classifiers:
#        for fr_method, fr_num in feature_reduction:
#
#            print("=======")
#            print(clf_method, fr_method, fr_num)
#            print("=======")
#            #DONE Print the results! This is applicable to both RF and ConvNet easily as RF supports this through sklearn, ConvNet supports this through skorch.
#
#            x = Quantification.process_all(pcap_filename = tssfilename, interactions = traces, 
#                quant_mode = clf_method, feature_reduction = fr_method, alignment = False, num_reduced_features = fr_num)
classifiers = ['convnn-classifier', 'fcnn-classifier', 'nb-classifier', 'knn-classifier','rf-classifier', 'convnn-classifier', ] #rnn-classifier (Not testing RNN right now.)
feature_reduction = [(None, 5), ('lda', 5), ('lda',10), ('pca',100), ('ranking', 5), ('ranking',10)] #PCA only reduces it to maximum possible number of independent features right now like the experiments in the paper, so the number doesn't mean anything for PCA.

traces = None

run_related = True 
run_gs = True
run_sa = True

#l = ['grpc_ac_device1.tss', 'grpc_ac_server1.tss', 'awsiot_lock_client1.tss', 'awsiot_lock_device1.tss']

l = ['awsiot_stove_client1.tss', 'awsiot_stove_device1.tss', 'awsiot_ac_client1.tss', 'awsiot_ac_device1.tss', 'awsiot_lock_client1.tss', 'awsiot_lock_device1.tss',  
    'grpc_ac_device1.tss', 'grpc_ac_server1.tss', 'grpc_cctv1.tss', 'grpc_stove_device1.tss', 'grpc_stove_server1.tss', 'grpc_switch1.tss']
#l = ['awsiot_ac_client1.tss', 'awsiot_ac_device1.tss', 'awsiot_lock_client1.tss', 'grpc_cctv1.tss']
#l = ['grpc_switch1.tss', 'grpc_cctv1.tss', 'awsiot_stove_client1.tss', 'awsiot_stove_device1.tss', 'awsiot_ac_client1.tss',]

very_start_time = time.time()
weights_list =  [(0.01, 1.0), (0.05, 1.0), (0.1, 1.0), (0.5, 1.0), (1.0, 1.0), (1.0, 0.5), (1.0, 0.1), (1.0, 0.05), (1.0, 0.01)]
    
print("Testing Event identification on Chaofan's Benchmark Dataset.")
for fn in l:
    for time_injection in [True, False]:
        print("RUNNING EXPS w/ TIME DELAY INJECTION = {}".format(time_injection))
        filename = "./UCSB_Traces/" + fn
        traces1 = Utils.parsetssfiles(filename)
        traces2 = Utils.parsetssfiles(filename[:-5] + '2.tss')
        traces = traces1[1:] + traces2[1:]
        print("")
        print("")
        print("*"*50)
        print("Analyzing {}, number of traces: {}".format(fn, len(traces)))
        print("*"*50)
        traces_ = copy.deepcopy(traces)
        
        x = Quantification.generate_defense(traces, filename, weights_list, run_related = run_related, run_gs = run_gs, run_sa = run_sa, time_injection=time_injection)
        print("TOTAL ELAPSED TIME: {:.3f} seconds".format(time.time() - very_start_time))

print("Testing Device identification on combined UNSW dataset.")
if True:
    #The folder containing the datasets.
    folder = "./Pinheiro_device_benchmark_traces/"

    #Maps the MAC address of an IoT device to a label (integer).
    devices ={"d0:52:a8:00:67:5e":1,"44:65:0d:56:cc:d3":2,"70:ee:50:18:34:43":3,"f4:f2:6d:93:51:f1":4,"00:16:6c:ab:6b:88":5,"30:8c:fb:2f:e4:b2":6,"00:62:6e:51:27:2e":7,"00:24:e4:11:18:a8":8,"ec:1a:59:79:f4:89":9,"50:c7:bf:00:56:39":10,"74:c6:3b:29:d7:1d":11,"ec:1a:59:83:28:11":12,"18:b4:30:25:be:e4":13,"70:ee:50:03:b8:ac":14,"00:24:e4:1b:6f:96":15,"74:6a:89:00:2e:25":16,"00:24:e4:20:28:c6":17,"d0:73:d5:01:83:08":18,"18:b7:9e:02:20:44":19,"e0:76:d0:33:bb:85":20,"70:5a:0f:e4:9b:c0":21}

    for ind, f in enumerate(os.listdir(folder)):
        traces = []
        print(str(f))
        print(folder+str(f))
        df = pd.read_csv(folder+str(f))[["eth.src", "TIME", "Size", "IP.src", "IP.dst", "port.src", "port.dst"]]

        #Replaces the MAC address of IoT devices with labels.
        for d in devices:
            df["eth.src"] = df["eth.src"].replace(d,devices[str(d)])

        #Extracts IoT devices from the original dataset.
        df = df[df['eth.src'].astype(str).str.isdigit()]

        #Groups packets into one-second windows for each IoT device.
        g = df.groupby(by=["TIME","eth.src"])

        for (el1, el2) in g:
            interaction = []
            #print("Element Header", el1)
            #print("Element Tail", el2)
            #p_list = el2.to_numpy() #Secret, Time, Size, IP.src, IP.dst, port.src, port.dst
            p_list = el2.values
            sec = p_list[0][0]
            src = p_list[0][3]
            dst = p_list[0][4]
            packet_load = 'INTERACTION_{}'.format(sec)
            p = Packet(src, dst, 55555, 55555, packet_load, len(packet_load), 0.0, 'M')
            interaction.append(p)
            for el in p_list:
                #sec   = el[0]
                time_  = el[1]
                size  = el[2]
                src   = el[3]
                dst   = el[4]
                sport = el[5]
                dport = el[6]
                p = Packet(src, dst, sport, dport, '', size, 0.0, '') #src, dst, sport, dport, load, size, time, flags)
                interaction.append(p)
            #print(len(interaction))
            traces.append(interaction)

        print("Len Traces", len(traces))
        print("")
        print("")
        print("*"*50)
        print("Analyzing {}, number of traces: {}".format(str(f), len(traces)))
        print("*"*50)
        x = Quantification.generate_defense(traces, str(f), weights_list, run_related = run_related, run_gs = run_gs, run_sa = run_sa, time_injection=False)

    very_end_time = time.time()
    print("TOTAL RUNTIME: {:.3f} seconds".format(very_end_time - very_start_time))
    print("END")
    #for clf_method in classifiers:
    #    for fr_method, fr_num in feature_reduction:
#
    #        print("=======")
    #        print(clf_method, fr_method, fr_num)
    #        print("=======")
    #        #DONE Print the results! This is applicable to both RF and ConvNet easily as RF supports this through sklearn, ConvNet supports this through skorch.
#
    #        x = Quantification.process_all(pcap_filename = f, interactions = traces, 
    #            quant_mode = clf_method, feature_reduction = fr_method, alignment = False, num_reduced_features = fr_num)

#Computes the statistical features for each one-second window and saves it to temporary CSV files.
#g.mean().to_csv("length_avg.csv",sep=",")
#g.sum().to_csv("length_sum.csv",sep=",")
#g.std().to_csv("length_std.csv",sep=",")

#Creates a new data frame to store the statistical features.
#df_final = pd.DataFrame()

#Populates the new data frame with statistical features and labels.
#df_final["avg"] = pd.read_csv("length_avg.csv")["Size"]
#df_final["n_bytes"] = pd.read_csv("length_sum.csv")["Size"]
#df_final["std"] = pd.read_csv("length_std.csv")["Size"]
#df_final["label"] = pd.read_csv("length_avg.csv")["eth.src"]

#Discard NaN values.
#df_final = df_final.dropna()

#Save the statistical features to a new CSV file
#df_final.to_csv(str(f)+"_statistics.csv",sep=",",mode='a',index=False,header=False)

#TODO Use Pinheiro's preprocessing tool, export the iot device identification code.




#if False:
#    #TODO Convert X and y to numpy arrays
#    skf = StratifiedKFold(n_splits=10)
#    for train_index, test_index in skf.split(X,y):
#        X_train, X_test = X[train_index], X[test_index]
#        y_train, y_test = y[train_index], y[test_index]
#
#        clf.fit(X_train, y_train)
#        y_rfc = clf.predict(X_test)
#
#        from sklearn.metrics import accuracy_score, recall_score, f1_score
#        from imblearn.metrics import geometric_mean_score, specificity_score
#
#        metrics['RandomForestClassifier'][0].append(accuracy_score(y_test, y_rfc))
#        metrics['RandomForestClassifier'][1].append(recall_score(y_test, y_rfc, average="micro"))
#        metrics['RandomForestClassifier'][2].append(f1_score(y_test, y_rfc, average="micro"))
#        metrics['RandomForestClassifier'][3].append(specificity_score(y_test, y_rfc, average="micro"))
#        metrics['RandomForestClassifier'][4].append(geometric_mean_score(y_test, y_rfc, average="micro"))
