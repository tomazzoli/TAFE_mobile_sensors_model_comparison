from commonFunctions import *

tipoSplit = "Shuffle"
bseName= "mieidaticonpredizione_epochs_400_timesteps_5_"

def splitTrainTest(dataframe,quota):
    test_size = 1- quota
    dfTrain, dfTest = train_test_split(dataframe, test_size=test_size)
    return (dfTrain,dfTest)


if __name__ == '__main__':
    outFileName =DIR_ESITI + os.path.sep + bseName + tipoSplit +".csv"
    for sensore in range(1, 8):
        filename = DIR_DATI + os.path.sep + "dataset_poi_"+ str(sensore)+".csv"
        dfMax = pd.read_csv(filename)
        try:
            dfOutConPred = pd.read_csv(outFileName)
        except:
            dfOutConPred = pd.DataFrame()
        dfTrain, dfTest = splitTrainTest(dfMax, 0.8)

        elab = elaboraSingoloSensore(dfMax,epochs = 400,dfTrain=dfTrain,dfTest=dfTest,sensore=sensore,batch_size=64)

        dfQuestoSensoreConPred = elab.get("dfOutConPred")
        nomeColTarget = elab.get("nomeColTarget")
        nomeColYTEST = elab.get("nomeColYTEST")
        nomeColPredicted = elab.get("nomeColPredicted")
        nomeColEvja = elab.get("nomeColEvja")

        ColTarget = dfQuestoSensoreConPred[nomeColTarget]
        ColYTEST = dfQuestoSensoreConPred[nomeColYTEST]
        ColPredicted = dfQuestoSensoreConPred[nomeColPredicted]
        ColEvja = dfQuestoSensoreConPred[nomeColEvja]

        dfOutConPred[nomeColTarget]=ColTarget.values
        dfOutConPred[nomeColYTEST]=ColYTEST.values
        dfOutConPred[nomeColPredicted]=ColPredicted.values
        dfOutConPred[nomeColEvja] = ColEvja.values
        dfOutConPred.to_csv(outFileName)

    print('finito')