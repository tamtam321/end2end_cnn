import tensorflow as tf
import mne
import numpy as np
import copy
from tensorflow import keras
from tensorflow.python.keras import layers

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# bizonyos warning vagy üzenet elfolytására van
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# TF_GPU_ALLOCATOR=cuda_malloc_async

subject_numb = 109
record_numb = 14    # 12 task + 2 baseline

# Task mátrixszoknak a sorai nem egyeznek meg és emiatt nem lehet konkatenálni őket.
# Mindegyik mátrixnak csak első 21 sorát használom.
reshape_row_val = 21

subjects_physical_tasks = []  # Subject lista, alanyhoz tartozó tényleges fizikai task rekordokat tartalmazza.
subjects_imaginary_tasks = []  # Subject lista, alanyhoz tartozó elképzelt task rekordokat tartalmazza.
learning_data_set = []
test_data_set = []
label_learning = []
label_test = []
model = 0   # neurháló model

# Ideiglenesen ide gyűjtöm ki az alanyhoz tartozó eventeket, aztán
# ennek egy másolata fogja reprezentálni az adott alanyt és ezt
# hozzáadom valamelyik subjects listához.
# Következő alanynál, kiürítem és újra használom.
subject_dict = {}

# _________________________________________

# Taskokhoz tartozó különböző eventek azonosítására kell.

events_map_1 = dict(
    Rest=1,  # T0
    LFist=2,  # T1
    RFist=3  # T2
)

events_map_2 = dict(
    Rest=1,  # T0
    BFists=2,  # T1
    BFeet=3  # T2
)
# _________________________________________

# Ezzel adom meg a learning és a test sethez a labelt.
# A labelben az eventek számmal lesznek azonsítva.
task_dict = dict(
    Rest=0,
    LFist=1,
    RFist=2,
    BFists=3,
    BFeet=4
)

# A Confusion mátrix plot tengely címkézéséhez
class_names = ["Rest", "LFist", "RFist", "BFists", "BFeet"]


# subject_dict-hez feltöltöm a jelenlegi alany eventjeit.
def loadEventsToSub(raw, events, event_id, event_1, event_2, event_3):
    # Epochs extracted from a Raw instance.
    tmp_epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-1, tmax=5, baseline=None)  # -1 -> 5, számít hány másodpercet mérünk, mert azzal is romolhat az eredmény, ha akár 1s-et is kihagyunk.

    if subject_dict.get(event_1) is None:
        subject_dict[event_1] = tmp_epochs[event_1].get_data()
    else:
        subject_dict[event_1] = np.append(subject_dict[event_1], tmp_epochs[event_1].get_data(), 0)

    if subject_dict.get(event_2) is None:
        subject_dict[event_2] = tmp_epochs[event_2].get_data()
    else:
        subject_dict[event_2] = np.append(subject_dict[event_2], tmp_epochs[event_2].get_data(), 0)

    if subject_dict.get(event_3) is None:
        subject_dict[event_3] = tmp_epochs[event_3].get_data()
    else:
        subject_dict[event_3] = np.append(subject_dict[event_3], tmp_epochs[event_3].get_data(), 0)


# **
# Megadja az adott alany elképzelt cselekvéseinek rekordját.*#
def getSubjectImaginaryTasks(subject_idx):
    file_path = []  # Fájl útvonala.

    for record_idx in range(record_numb):
        file_path.append(f'files/S{subject_idx:03}/S{subject_idx:03}R{(record_idx + 1):02d}.edf')

    # Első két rekordot nem használjuk, azok csak baseline anyagok.
    # (Első elem indexe 0) 3. indextől kettesével lépkedve érem el a képzeletbeli taskokat.
    # Vagyis 3-13 közötti idexszek közül a páratlanok azok a képzeletbeli taskok.
    for s in range(3, record_numb, 2):

        # A 88, 89, 92 és 100-as azonosítójú subjecteket nem veszem figyelembe, mert azok más frekvencia
        # értékkel lettek rögzítve, mint a többi.
        if file_path[s][-10:-7] == '088' or \
                file_path[s][-10:-7] == '089' or file_path[s][-10:-7] == '092' or \
                file_path[s][-10:-7] == '100':

            return
        else:
            raw = mne.io.read_raw_edf(file_path[s])     # adat nyers formában
            tmp_events, _ = mne.events_from_annotations(raw)    # eventek

            # 4, 8 és 12-es id-jú rekord tartalmazza a rest, left, right fist eventeket.
            # A maradék, ami az 6, 10, 14-as id-jú rekord tartalmazza a rest, both fists, feet eventeket.
            # A file_path-ban megvizsgálom az id-ját a rekordnak.
            if file_path[s][-6:-4] == '04' or file_path[s][-6:-4] == '08' or \
                    file_path[s][-6:-4] == '12':

                # subject_dict-hez feltöltöm a jelenlegi alany eventjeit.
                loadEventsToSub(raw, tmp_events, events_map_1, "Rest", "LFist", "RFist")

            else:   # Rekord id 6, 10, 14 -> (rest, both fists, feet)
                loadEventsToSub(raw, tmp_events, events_map_2, "Rest", "BFists", "BFeet")

    # subject_dict másolatát tárolom el a subjects_imaginary_tasks listában.
    # A másolat egy subject-et reprezentál.
    tmp_dict = copy.deepcopy(subject_dict)
    subjects_imaginary_tasks.append(tmp_dict)
    subject_dict.clear()    # ürítem és újrahasználom


# **
# Megadja az adott alany tényleges fizikai cselekvéseinek rekordját.*#
def getSubjectPhysicalTasks(subject_idx):
    file_path = []

    for record_idx in range(record_numb):
        file_path.append(f'files/S{subject_idx:03}/S{subject_idx:03}R{(record_idx + 1):02d}.edf')

    # (0-tól nézem az indexszet)
    # Mint a getSubjectImaginaryTasks, csak a 2. indexsztől lépegetek kettőket.
    # 2-13 között a páros indexszűek, azok a fizikai taskok.
    for s in range(2, record_numb, 2):
        if file_path[s][-10:-7] == '088' or \
                file_path[s][-10:-7] == '089' or file_path[s][-10:-7] == '092' or \
                file_path[s][-10:-7] == '100':

            return
        else:
            raw = mne.io.read_raw_edf(file_path[s])
            tmp_events, _ = mne.events_from_annotations(raw)

            # Rekord id 3, 7, 11 -> (rest, left, right fist)
            if file_path[s][-6:-4] == '03' or file_path[s][-6:-4] == '07' or \
                    file_path[s][-6:-4] == '11':

                loadEventsToSub(raw, tmp_events, events_map_1, "Rest", "LFist", "RFist")

            else:  # Rekord id 5, 9, 13 -> (rest, both fists, feet)
                loadEventsToSub(raw, tmp_events, events_map_2, "Rest", "BFists", "BFeet")

    tmp_dict = copy.deepcopy(subject_dict)
    subjects_physical_tasks.append(tmp_dict)
    subject_dict.clear()


# Van egy listám dictionarykkel.
# Az összes dictionary-nek lekérem az értékét egy listába és
# azzal térek vissza.
def getValFromDict(dictionaries):
    tmp_value_list = []

    for data in dictionaries:
        for key, value in data.items():
            tmp_value_list.append(value)

    return tmp_value_list


# Újraformázom a mátrixszokat.
# Nem tudom konkatenálni a mátrixszokat, mert nem ugyanolyan a soruk száma.
# Ezért mindegyiket ugyanolyan sorszámra állítom, a legkisebb alapján.

# TO DO:
# 1) reshape_row_val -> még mindig 21 vagy már más?
#   meg kéne nézni a debugban, hogy most mennyi a legkisebb soru mátrix és
#   ahhoz igaztani a többit is. Talán megnövelné a pontosságot, mert több
#   adatot fog így tárolni, ha az érték nagyobb, mint most a jelenlegi, ami a 21.

def reshapeMatrix(matrices):
    tmp_matrix_list = []

    for m in matrices:
        tmp_matrix_list.append(m[0:reshape_row_val, :, :])

    return tmp_matrix_list


# Mátrix értékeit normalizálom.
# Azért kell, mert a dataset értékei túl kicsik és azzal nem fog tudni dolgozni a háló.
# Normalizálás után az értékek, megfelelően nagyok lesznek.
def normalizeMatrixVal(matrix):
    matrix_values_mean_np = np.mean(matrix)
    matrix_std_dev_np = np.std(matrix)  # Mátrix értékeinek a szórása (Standard deviation)
    matrix_values_normalized_np = (matrix - matrix_values_mean_np) / matrix_std_dev_np

    return matrix_values_normalized_np


# normalizálás -> adott értékből kivonom az átlagot és azt leosztom a szórással.
# Learning és a test datasetben túl kicsik az értékek és normalizálni kell, hogy megfelelően nagy legyen,
# hogy az eredmény jó legyen. Újraformázom a mátrix első paraméterét, hogy egységes legyen konkatenálás során.
def reshapeAndNormalize(learning_data, test_data):
    tmp_learning_np = np.array(learning_data)
    tmp_values_learning = getValFromDict(tmp_learning_np)
    tmp_matrices_learning = reshapeMatrix(tmp_values_learning)
    tmp_matrix_learning = np.concatenate(tmp_matrices_learning, axis=0)  # Learning adatokat összefűzöm.
    learning_set_np_normalized = normalizeMatrixVal(tmp_matrix_learning)

    tmp_test_np = np.array(test_data)
    tmp_values_test = getValFromDict(tmp_test_np)
    tmp_matrices_test = reshapeMatrix(tmp_values_test)
    tmp_matrix_test = np.concatenate(tmp_matrices_test, axis=0)  # Test adatokat összefűzöm.
    test_set_np_normalized = normalizeMatrixVal(tmp_matrix_test)

    return learning_set_np_normalized, test_set_np_normalized


#   Label létrehozása.
def createLabelList(dictionaries):
    list_ = []

    for dictionary in dictionaries:
        for key, value in dictionary.items():

            # Mivel az eventhez tartozó mátrix 21 soros, ezért magát a labelt
            # annyiszor fogom beletenni.
            for i in range(reshape_row_val):
                list_.append(task_dict[key])

    list_ = np.array(list_)
    return list_


# val érték benne van az intervallumban?
def isInInterval(min_, max_, val):
    if min_ <= val < max_:
        return True

    return False


# **
# Szétválagotam a beolvasott alanyokat learning és test dataset-re.*#
def getLearningTestDataSet(min_, max_):
    learning_data_set_tmp = []
    test_data_set_tmp = []
    global label_learning
    global label_test

    # **
    # 105 alanyt, 80% learning 20% test dataset-re bontom.
    # [min_; max_) értékek közti intervallum adja meg, hogy melyik alany tartozik majd
    # a test dataset-be.*
    counter_tmp = 0
    for subject in subjects_imaginary_tasks:
        if isInInterval(min_, max_, counter_tmp):
            test_data_set_tmp.append(subject)
        else:
            learning_data_set_tmp.append(subject)

        counter_tmp += 1

    label_learning = createLabelList(learning_data_set_tmp)
    label_test = createLabelList(test_data_set_tmp)

    learning_data_set_tmp, test_data_set_tmp = reshapeAndNormalize(learning_data_set_tmp, test_data_set_tmp)

    return (learning_data_set_tmp, label_learning), (test_data_set_tmp, label_test)


def confusionMatrix(test_label, test_data_pred):
    y_test = test_label
    y_pred = test_data_pred
    y_classes = np.argmax(y_pred, axis=1)
    mat = confusion_matrix(y_test, y_classes)
    plot_confusion_matrix(conf_mat=mat, figsize=(8, 8), class_names=class_names, show_normed=True)
    plt.show()


# tanításhoz
batch_size = 16
epochs = 30


# **
# Háló tanítás*#
def startNN():

    # Lekérem az alanyok MI adatát.
    for i in range(50):
        getSubjectImaginaryTasks(i + 1)

    block = 0
    global learning_data_set
    global test_data_set
    global model

    accuracies = []  # Teszt sikeressége.

    test_labels_np = []
    test_data_np_for_predict = []

    # Crossvalidation ->  80% learning, 20% test
    # 5 block
    while block < 5:
        lower_bound = block * 10
        upper_bound = block * 10 + 10

        learning_data_set, test_data_set = getLearningTestDataSet(lower_bound, upper_bound)

        # accuracies.clear()

        # model
        # data set szerkezet -> mátrix: (adat darabszám x csatornaszám x idő)
        # NEEG -> number of channels
        # N -> length of the input (az idő fogja meghatározni)
        # 1 epoch -> 1 eventet tartalmaz, 4s, 64 csatornán figyelve, input hossza 321 -> 4/321 ~ 12ms
        # Első layernél megadjuk, hogy mekkora lesz az input (64x321x1) és onnantól majd első layer outputja (64x321x40) megy tovább a kövi layer inputjaként ...stb.
        model = keras.models.Sequential()
        model.add(layers.Conv2D(40, (1, 30), strides=(1, 1), padding="same",
                                activation="relu", input_shape=(64, 321, 1)))  # temporal convolution (64, 321, 1) - (NEEG, N, 1)
        model.add(layers.Conv2D(40, (64, 1), strides=(1, 1), padding="valid", activation="relu"))  # spatial conv # input_shape=(64, 321, 40) - (NEEG, N, 40)
        model.add(layers.AvgPool2D(pool_size=(1, 15), strides=(1, 1), padding="valid"))  # pooling layer # input_shape=(1, 321, 40) - (1, N, 40)
        model.add(layers.Flatten())  # flatten
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(5))

        # loss és optimizer
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optim = keras.optimizers.Adam(learning_rate=0.001)
        metrics = ["accuracy"]

        model.compile(optimizer=optim, loss=loss, metrics=metrics)

        # tanítás
        model.fit(np.expand_dims(learning_data_set[0], axis=-1), np.expand_dims(learning_data_set[1], axis=-1), epochs=epochs,
                  batch_size=batch_size, verbose=2, validation_split=0.2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)], shuffle=True)

        # model.fit(np.expand_dims(learning_data_set[0], axis=-1), np.expand_dims(learning_data_set[1], axis=-1),
        #           epochs=epochs,
        #           batch_size=batch_size, verbose=2, validation_split=0.2,
        #           shuffle=True)

        # kiértékelés
        _, acc = model.evaluate(np.expand_dims(test_data_set[0], axis=-1),
                                np.expand_dims(test_data_set[1], axis=-1), batch_size=batch_size, verbose=2)
        accuracies.append(acc)

        if block == 0:
            test_labels_np = np.expand_dims(test_data_set[1], axis=-1)
            test_data_np_for_predict = model.predict(np.expand_dims(test_data_set[0], axis=-1))
        else:
            test_labels_np = np.append(test_labels_np, np.expand_dims(test_data_set[1], axis=-1), axis=0)
            test_data_np_for_predict = np.append(test_data_np_for_predict, model.predict(np.expand_dims(test_data_set[0], axis=-1)), axis=0)

        block += 1

    test_labels_concatenated = np.concatenate(test_labels_np, axis=0)

    # Confusion Matrix
    confusionMatrix(test_labels_concatenated,
                    test_data_np_for_predict)

    # végső teszt eredmény
    test_accuracy = np.mean(accuracies)

    print("!!!!!!!!!!!!!!!_TESZT HALMAZOK PONTOSSÁGA_!!!!!!!!!!!!: ", accuracies)
    print("!!!!!!!!!!!!!!!_EGÉSZ TESZT ÁTLAG_!!!!!!!!!!!!: ", test_accuracy)


# Háló indítása
startNN()
# print(tf.version.VERSION)
print("Kész")
