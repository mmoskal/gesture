import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

const STEADY_TOLERANCE = 2
const MIN_GESTURE_LEN = 20
const MAX_GESTURE_ACC = 5
const NUM_SAMPLES = 50;
const NUM_DIM = 3;
const IMAGE_CHANNELS = 1;

const BATCH_SIZE = 32;
const BATCHES_PER_EPOCH = 10;
const NUM_EPOCHS = 60;

// find data -name \*.csv
const fileNames = `
data/michal/punch.csv
data/michal/right.csv
data/michal/left.csv
data/michal/noise.csv
data/michal/noise1.csv
data/ira/punch2.csv
data/ira/right1.csv
data/ira/left0.csv
data/ira/noise0.csv
`

const classNames = ['noise', 'punch', 'left', 'right'];

type SMap<T> = {
    [x: string]: T;
}
function median(arr: number[]) {
    arr.sort((a, b) => a - b)
    return arr[arr.length >> 1]
}
function dist(a: number[], b: number[]) {
    let sum = 0
    if (a.length != b.length)
        throw new Error("wrong size")
    for (let i = 0; i < a.length; i++)
        sum += Math.abs(a[i] - b[i])
    return sum
}

interface Range {
    id: number;
    preStart: number;
    start: number;
    stop: number;
    postStop: number;
}

class DataProvider {
    private samples: number[][]
    ranges: Range[]

    constructor(private csvurl: string, private id: number = null) {
    }

    dataset() {
        const self = this
        function* gen() {
            for (let i = 0; i < BATCHES_PER_EPOCH; ++i)
                yield self.getBatch(BATCH_SIZE)
        }
        return tf.data.generator(gen)
    }

    get className() {
        if (this.id == null)
            return "???"
        return classNames[this.id]
    }

    private noiseRanges() {
        const sampleLen = NUM_SAMPLES
        const len = sampleLen + (sampleLen >> 1)
        const midlen = sampleLen >> 1
        this.ranges = []
        for (let off = 0; off + len < this.samples.length; off += len) {
            this.ranges.push({
                id: this.id,
                preStart: off,
                start: off + (len - midlen >> 1),
                stop: off + (len + midlen >> 1),
                postStop: off + len,
            })
        }
        console.log("noise", this.ranges)
    }

    async load() {
        console.log("loading " + this.csvurl)
        const csv = tf.data.csv(this.csvurl)
        const buckets: SMap<number[][]> = {}
        const allsamples: number[][] = []
        this.samples = []
        for (const obj of await csv.toArray()) {
            const vals = Object.values(obj as any).slice(1) as number[]
            const bucketId = vals.map(v => Math.round(v * 5)).join(",")
            if (!buckets[bucketId])
                buckets[bucketId] = []
            buckets[bucketId].push(vals)
            allsamples.push(vals)
            this.samples.push(vals.slice(0))
        }
        if (/noise/.test(this.csvurl)) {
            this.noiseRanges()
            return
        }
        const bids = Object.keys(buckets)
        bids.sort((a, b) => buckets[b].length - buckets[a].length)
        const topnum = buckets[bids[0]].length
        const avgbuckets = bids.slice(0, 6).map(bid => buckets[bid]).filter(x => x.length > (topnum / 10))
        const avgsamples: number[][] = []
        avgbuckets.forEach(a => a.forEach(b => avgsamples.push(b)))
        const med = [0, 1, 2].map(idx => median(avgsamples.map(a => a[idx])))
        console.log("steady:", med)
        const distances = allsamples.map(s => dist(med, s))
        const meddist = median(distances)
        const cutoff = meddist * STEADY_TOLERANCE
        console.log("cutoff:", cutoff, "in cutoff %:", distances.filter(d => d < cutoff).length * 100 / distances.length)

        let acc = 0
        let lastbeg = -1
        let idx = 0
        let prevEnd = 0
        this.ranges = []
        for (const sample of allsamples) {
            const d = dist(med, sample)
            sample.push(d > cutoff ? -1 : -2)
            if (d > cutoff) {
                acc++
                if (lastbeg == -1)
                    lastbeg = idx
            } else {
                if (acc) {
                    acc--
                    if (!acc && lastbeg != -1) {
                        const len = idx - lastbeg
                        if (len > MIN_GESTURE_LEN) {
                            for (let i = lastbeg - 3; i <= idx; ++i)
                                allsamples[i][3] += 3
                            this.ranges.push({
                                id: this.id,
                                preStart: prevEnd,
                                start: Math.max(lastbeg - 3, 0),
                                stop: idx,
                                postStop: -1
                            })
                        }
                        lastbeg = -1
                    }
                }
            }
            acc = Math.min(MAX_GESTURE_ACC, acc)
            idx++
        }

        for (let i = 1; i < this.ranges.length; ++i) {
            this.ranges[i - 1].postStop = this.ranges[i].start - 1
        }
        this.ranges[this.ranges.length - 1].postStop = allsamples.length - 1
        console.log(this.ranges)

        for (let i = 0; i < allsamples.length; i += 400) {
            //showDet(allsamples.slice(i, i + 400), this.className + " " + i)
        }

        function showDet(allsamples: number[][], name: string) {
            const series = ['X', 'Y', 'Z', 'G'];
            const xdata = {
                values: [0, 1, 2, 3].map(n =>
                    allsamples.map((s, i) => ({ x: i, y: s[n] }))), series
            }
            const surface = { name, tab: 'Charts' };
            tfvis.render.linechart(surface, xdata);
        }
    }

    append(other: DataProvider) {
        let off = 0
        if (!this.samples || !this.samples.length)
            this.samples = other.samples
        else {
            off = this.samples.length
            for (const s of other.samples)
                this.samples.push(s)
        }
        if (!this.ranges) this.ranges = []
        for (const r of other.ranges) {
            this.ranges.push({
                id: r.id,
                preStart: r.preStart + off,
                start: r.start + off,
                stop: r.stop + off,
                postStop: r.postStop + off,
            })
        }
    }

    private copy(other: DataProvider) {
        this.samples = other.samples
    }

    split(firstFrac: number): [DataProvider, DataProvider] {
        const cutoff = Math.round(firstFrac * this.ranges.length)
        const r0 = new DataProvider(this.csvurl, this.id)
        r0.copy(this)
        r0.ranges = this.ranges.slice(0, cutoff)
        const r1 = new DataProvider(this.csvurl, this.id)
        r1.copy(this)
        r1.ranges = this.ranges.slice(cutoff)
        return [r0, r1]
    }


    filterRanges() {
        const l0 = this.ranges.length
        this.ranges = this.ranges.filter(r => r.stop - r.start < NUM_SAMPLES - 2)
        const l1 = this.ranges.length
        let drop = l0 - l1
        if (drop)
            console.log(this.csvurl, `drop ${drop} too long`)
        this.ranges = this.ranges.filter(r => r.postStop - r.preStart > NUM_SAMPLES + 2)
        const l2 = this.ranges.length
        drop = l1 - l2
        if (drop)
            console.log(this.csvurl, `drop ${drop} with too little wiggle`)
        permute(this.ranges)
    }

    private rangeSamples(r: Range) {
        const len = r.start - r.preStart
        const off = r.preStart + ((Math.random() * len) | 0)
        return this.samples.slice(off, off + NUM_SAMPLES)
    }

    private rangeLabels(rng: Range) {
        return classNames.map((_, i) => rng.id == i ? 1 : 0)
    }

    getBatch(batchSize: number) {
        const ranges: Range[] = []
        for (let i = 0; i < batchSize; ++i)
            ranges.push(pickRandom(this.ranges))
        return tf.tidy(() => ({
            xs: tf.tensor(ranges.map(r => this.rangeSamples(r)))
                .reshape([batchSize, NUM_SAMPLES, NUM_DIM, IMAGE_CHANNELS]),
            ys: tf.tensor(ranges.map(r => this.rangeLabels(r)))
        }))
    }
}

async function run() {
    const datasets: DataProvider[] = []
    for (const fn of fileNames.split(/\n/).map(s => s.trim())) {
        if (!fn) continue
        const idx = classNames.findIndex(cl => fn.indexOf(cl) >= 0)
        const d = new DataProvider(fn, idx)
        datasets.push(d)
    }

    let lens: number[] = []
    for (const d of datasets) {
        await d.load()
        for (const r of d.ranges) {
            lens.push(r.stop - r.start)
        }
    }
    console.log(lens)
    console.log("median len: " + median(lens))
    console.log("len 50+: " + lens.filter(l => l > NUM_SAMPLES).length)
    for (const d of datasets) {
        await d.filterRanges()
    }

    const trainData = new DataProvider("train")
    const testData = new DataProvider("test")

    for (const d of datasets) {
        const [test, train] = d.split(0.2)
        trainData.append(train)
        testData.append(test)
        console.log(d.className, test.ranges.length, train.ranges.length)
    }
    trainData.filterRanges()
    testData.filterRanges()


    const model = getModel();
    tfvis.show.modelSummary({ name: 'Model Architecture' }, model);

    await train(model, trainData, testData);

    await showAccuracy(model, trainData)
    await showConfusion(model, trainData)
}

document.addEventListener('DOMContentLoaded', run);

function doPrediction(model: tf.LayersModel, data: DataProvider, testDataSize = 500) {
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.getBatch(testDataSize);
    const testxs = testData.xs // .reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
    const labels = testData.ys.argMax(-1) as tf.Tensor1D;
    const preds = (model.predict(testxs) as tf.Tensor1D).argMax(-1) as tf.Tensor1D;
    testxs.dispose();
    return [preds, labels];
}


async function showAccuracy(model: tf.LayersModel, data: DataProvider) {
    const [preds, labels] = doPrediction(model, data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = { name: 'Accuracy', tab: 'Evaluation' };
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

    labels.dispose();
}

async function showConfusion(model: tf.LayersModel, data: DataProvider) {
    const [preds, labels] = doPrediction(model, data);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
    tfvis.render.confusionMatrix(
        container, { values: confusionMatrix, tickLabels: classNames });

    labels.dispose();
}


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

function pickRandom<T>(arr: T[]) {
    return arr[Math.random() * arr.length | 0]
}

async function train(model: tf.LayersModel, train: DataProvider, test: DataProvider) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
        name: 'Model Training', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    return model.fitDataset(train.dataset(), {
        validationData: test.dataset(),
        epochs: NUM_EPOCHS,
        callbacks: fitCallbacks
    });
}