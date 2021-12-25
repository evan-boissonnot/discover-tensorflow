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

document.getElementById('btnLoad').addEventListener('click', run);