const lucide = window.lucide;

// getVideo function
let track;
async function getVideo(facingMode) {
	const mediaStream = await navigator.mediaDevices.getUserMedia({
		video: { facingMode: facingMode },
	});
	document.getElementById('video').srcObject = mediaStream;
	document.getElementById('video').play();
	track = mediaStream.getVideoTracks()[0];
}

// flashlight button
let torch = false;
document.getElementById('flashlight').addEventListener('click', () => {
	track
		.applyConstraints({
			advanced: [{ torch: !torch }],
		})
		.then(() => {
			torch = !torch;

			document.getElementById('flashlight').innerHTML = `<i data-lucide="${
				torch ? 'flashlight-off' : 'flashlight'
			}" class="button"></i>`;
			lucide.createIcons();
		});
});

// flip camera button
let facingMode = 'environment';
document.getElementById('flip').addEventListener('click', () => {
	const newFacingMode = facingMode == 'environment' ? 'user' : 'environment';
	getVideo(newFacingMode).then(() => {
		facingMode = newFacingMode;
	});
});

// menu button
document.getElementById('menu').addEventListener('click', () => {
	// TODO: add menu button functionality
});

// initialization
lucide.createIcons();
getVideo('environment');
