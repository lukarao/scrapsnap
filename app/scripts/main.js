import {
	createIcons,
	Flashlight,
	FlashlightOff,
	RefreshCw,
	Menu,
} from 'https://esm.sh/lucide';

// camera functionality
const video = document.getElementById('video');
let track;

async function getCamera(facingMode) {
	const mediaStream = await navigator.mediaDevices.getUserMedia({
		video: { facingMode: facingMode },
	});
	video.srcObject = mediaStream;
	video.play();
	track = mediaStream.getVideoTracks()[0];
}

// flashlight button
const flashlight = document.getElementById('flashlight');
let torch = false;

function updateFlashlightButton() {
	if (torch) {
		flashlight.innerHTML =
			'<i data-lucide="flashlight-off" class="button"></i>';
		createIcons({ icons: { FlashlightOff } });
	} else {
		flashlight.innerHTML = '<i data-lucide="flashlight" class="button"></i>';
		createIcons({ icons: { Flashlight } });
	}
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

// menu button
document.getElementById('menu').addEventListener('click', () => {
	// TODO: add menu button functionality
});

// initialization
createIcons({ icons: { Flashlight, RefreshCw, Menu } });
getVideo(facingMode);
