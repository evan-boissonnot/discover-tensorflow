import { MnistData } from './data.js';

async function showExamples(data, nbResults) {
    const surface = tfvis.visor().surface({ name: 'Input data', tab: 'Data' });

    const examples = data.nextTestBatch( parseInt(nbResults));
    const numExamples = examples.xs.shape[0];

    for (let index = 0; index < numExamples; index++) {
        const imageTensor = tf.tidy(
            () => {
                return examples.xs
                               .slice([index, 0], [1, examples.xs.shape[1]])
                               .reshape([28, 28, 1]);
            }
        );
        
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;

        canvas.style = 'margin: 10px';

        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }
}

async function run() {
    const data = new MnistData();
    await data.load();
    await showExamples(data, document.getElementById('nbImages').value);
}

function getModel() {
    const model = tf.sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    model.add(
        tf.layers.maxPooling2d({ 
            poolSize: [2, 2],
            strides: [2, 2]
        })
    );

    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    model.add(
        tf.layers.maxPooling2d({ 
            poolSize: [2, 2],
            strides: [2, 2]
        })
    );

    model.add(tf.layers.flatten());

    const NUM_OUPUT_CLASSES = 10; // from 0 to 9 digit
    model.add(
        tf.layers.dense({
            units: NUM_OUPUT_CLASSES,
            kernelInitializer: 'varianceScaling',
            activation: 'softmax'
        })
    );

    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

document.getElementById('btnLoad').addEventListener('click', run);