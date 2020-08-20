import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'


interface Gesture {
    label: string;
    samples: number[][];
}

let gestures: Gesture[]

function drawGesture(gesture: Gesture, name: string) {
    const series = ['X', 'Y', 'Z'];
    const xdata = {
        values: [0, 1, 2].map(n =>
            gesture.samples.map((s, i) => ({ x: i, y: s[n] }))), series
    }
    const surface = { name: name || gesture.label, tab: 'Charts' };
    tfvis.render.linechart(surface, xdata);
}

async function showExamples() {
    for (let i = 0; i < gestures.length; ++i) {
        drawGesture(gestures[i], gestures[i].label + " " + i)
    }
}

async function run() {
    const resp = await fetch(new Request("data.json"))
    gestures = await resp.json()
    // await showExamples();

    const model = getModel();
    tfvis.show.modelSummary({ name: 'Model Architecture' }, model);

    await train(model);
}

document.addEventListener('DOMContentLoaded', run);

const NUM_SAMPLES = 50;
const NUM_DIM = 3;
const IMAGE_CHANNELS = 1;

const classNames = ['noise', 'punch', 'left', 'right'];

/*
function doPrediction(model, data, testDataSize = 500) {
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);

    testxs.dispose();
    return [preds, labels];
}


async function showAccuracy(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = { name: 'Accuracy', tab: 'Evaluation' };
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

    labels.dispose();
}

async function showConfusion(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
    tfvis.render.confusionMatrix(
        container, { values: confusionMatrix }, classNames);

    labels.dispose();
}
*/

function getModel() {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [NUM_SAMPLES, NUM_DIM, IMAGE_CHANNELS],
        kernelSize: [4, 3],
        filters: 8,
        strides: 1,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({ poolSize: [3, 3], strides: [3, 3] }));
    model.add(tf.layers.dropout({ rate: 0.1 }));

    model.add(tf.layers.conv2d({
        kernelSize: [4, 1],
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 1], strides: [2, 1] }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units: classNames.length,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));


    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}

function permute<T>(arr: T[]) {
    for (let i = 0; i < arr.length; ++i) {
        const a = (Math.random() * arr.length) | 0
        const b = (Math.random() * arr.length) | 0
        const tmp = arr[a]
        arr[a] = arr[b]
        arr[b] = tmp
    }
}

async function train(model: tf.LayersModel) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
        name: 'Model Training', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 8;

    const trainData: Gesture[] = []
    const testData: Gesture[] = []
    for (const lbl of classNames) {
        const gg = gestures.filter(g => g.label == lbl)
        permute(gg)
        const idx = (gg.length / 4) | 0
        for (let i = 0; i < gg.length; ++i) {
            const s = gg[i].samples
            if (s.length < 0.95 * NUM_SAMPLES || s.length > 1.05 * NUM_SAMPLES)
                throw new Error("Bad length: " + s.length)
            while (s.length < NUM_SAMPLES)
                s.push(s[s.length - 1])
            gg[i].samples = s.slice(0, NUM_SAMPLES)
            if (i < idx) testData.push(gg[i])
            else trainData.push(gg[i])
        }
    }
    permute(trainData)
    permute(testData)

    function toTensors(gg: Gesture[]) {
        return [
            tf.tensor(gg.map(g => g.samples)).reshape([gg.length, NUM_SAMPLES, NUM_DIM, IMAGE_CHANNELS]),
            tf.tensor(gg.map(g => {
                const r = []
                for (const lbl of classNames)
                    r.push(lbl == g.label ? 1 : 0)
                return r
            }))
        ]
    }

    const [trainXs, trainYs] = tf.tidy(() => toTensors(trainData))
    const [testXs, testYs] = tf.tidy(() => toTensors(testData))

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 50,
        shuffle: true,
        callbacks: fitCallbacks
    });
}