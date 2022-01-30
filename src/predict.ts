import * as tf from "@tensorflow/tfjs-node"
import {Sequential} from "@tensorflow/tfjs-node"

class Data {

    readonly sizeMB: Array<number>;

    readonly timeSec: Array<number>;


    constructor(sizeMB: Array<number>, timeSec: Array<number>) {
        this.sizeMB = sizeMB;
        this.timeSec = timeSec;
    }

}


const trainData = new Data(
    [0.080, 9.000, 0.001, 0.100, 8.000,
        5.000, 0.100, 6.000, 0.050, 0.500,
        0.002, 2.000, 0.005, 10.00, 0.010,
        7.000, 6.000, 5.000, 1.000, 1.000],

    [0.135, 0.739, 0.067, 0.126, 0.646,
        0.435, 0.069, 0.497, 0.068, 0.116,
        0.070, 0.289, 0.076, 0.744, 0.083,
        0.560, 0.480, 0.399, 0.153, 0.149]
)

const testData = new Data(
    [5.000, 0.200, 0.001, 9.000, 0.002,
        0.020, 0.008, 4.000, 0.001, 1.000,
        0.005, 0.080, 0.800, 0.200, 0.050,
        7.000, 0.005, 0.002, 8.000, 0.008],

    [0.425, 0.098, 0.052, 0.686, 0.066,
        0.078, 0.070, 0.375, 0.058, 0.136,
        0.052, 0.063, 0.183, 0.087, 0.066,
        0.558, 0.066, 0.068, 0.610, 0.057]
)

// we can use tensor1d also
const trainTensors = {
    sizeMB: tf.tensor2d(trainData.sizeMB, [20, 1]),
    timeSec: tf.tensor2d(trainData.timeSec, [20, 1])
}

const testTensors = {
    sizeMB: tf.tensor2d(testData.sizeMB, [20, 1]),
    timeSec: tf.tensor2d(testData.timeSec, [20, 1])
}

// timeSec = kernel * sizeMB + bias

const model: Sequential = tf.sequential()
model.add(tf.layers.dense({units: 1, inputShape: [1]}))
model.compile({optimizer: "sgd", loss: "meanAbsoluteError"})
// sgd - stochastic gradient descent/стохастический градиентный спуск
/*
modelOutput = [1.1, 2.2, 3.3, 3.6]
targets =     [1.0, 2.0, 3.0, 4.0]
meanAbsoluteError = average([|1.1 - 1.0|, |2.2 - 2.0|,
                            |3.3 - 3.0|, |3.6 - 4.0|])
                  = average([0.1, 0.2, 0.3, 0.4])
                  = 0.25
less is better!
**/

console.log("Average loss for trainData.timeSec")
console.log(tf.mean(trainData.timeSec).toString())
console.log("------------------------------------")

console.log("Average loss for testData.timeSec")
// 'abs' value for difference between target value and our constant prediction in step above
console.log(tf.mean(tf.abs(tf.sub(testData.timeSec, 0.295))).toString())
console.log("------------------------------------")


const trainModel = async () => {
    await model.fit(trainTensors.sizeMB, trainTensors.timeSec, {epochs: 200})
    console.log(model.evaluate(testTensors.sizeMB, testTensors.timeSec).toString)
}

trainModel().then(() => {
    const smallFileMB = 1
    const bigFileMB = 100
    const hugeFileMB = 10000
    // const predict = model.predict(tf.tensor2d([[smallFileMB], [bigFileMB], [hugeFileMB]]))
    // because of inputShape: [1]
    const predict = model.predict(tf.tensor2d([smallFileMB, bigFileMB, hugeFileMB], [3,1]))
    console.log(predict.toString(true))
    console.log(predict)
})

