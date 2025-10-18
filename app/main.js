navigator.mediaDevices
	.getUserMedia({
		video: { facingMode: 'environment' },
	})
	.then((mediaStream) => {
		document.getElementById('video').srcObject = mediaStream;
	})
	.catch((error) => alert(error));
