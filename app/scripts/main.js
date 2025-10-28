import { updateIcons } from './icons.js';
import { processFrame, mlStop, mlResume } from './ml.js';

// camera functionality
const video = document.getElementById('video');
let track;

async function getCamera(facingMode) {
	const mediaStream = await navigator.mediaDevices.getUserMedia({
		video: { facingMode: facingMode },
	});
	video.srcObject = mediaStream;
	video.play();

	// initialize ml frame processing
	video.requestVideoFrameCallback(processFrame);

	track = mediaStream.getVideoTracks()[0];
}

// flashlight button
const flashlight = document.getElementById('flashlight');
let torch = false;

function updateFlashlightButton() {
	flashlight.innerHTML = `<i data-lucide="${
		torch ? 'flashlight-off' : 'flashlight'
	}" class="button"></i>`;
	updateIcons();
}

document.getElementById('flashlight').addEventListener('click', () => {
	track
		.applyConstraints({
			advanced: [{ torch: !torch }],
		})
		.then(() => {
			torch = !torch;
			updateFlashlightButton();
		});
});

// flip camera button
let facingMode = 'environment';

document.getElementById('flip').addEventListener('click', () => {
	const newFacingMode = facingMode == 'environment' ? 'user' : 'environment';
	getCamera(newFacingMode).then(() => {
		facingMode = newFacingMode;
		torch = false;
		updateFlashlightButton();
	});
});

// menu
const menu = document.getElementById('menu');

document.getElementById('menu-open').addEventListener('click', () => {
	mlStop();
	setTimeout(() => {
		menu.style.left = '0%';
	}, 100);
});

document.getElementById('menu-close').addEventListener('click', () => {
	menu.style.left = '100%';
	setTimeout(() => {
		mlResume();
	}, 500);
});

// initialization
updateIcons();
getCamera(facingMode);
