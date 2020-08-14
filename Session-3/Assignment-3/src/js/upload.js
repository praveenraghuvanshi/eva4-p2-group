function uploadAndClassifyImage(url){
	var fileInput = document.getElementById('resnet34FileUpload').files;
	if(!fileInput.length){
		return alert('Please choose a file to upload first');
	}
	
	var file = fileInput[0];
	var filename = file.name;
	
	var formData = new FormData();
	formData.append(filename, file);
	
	console.log(filename);
	console.log(url);
	
	$.ajax({
		async: true,
		crossDomain: true,
		method: 'POST',
		url: url,
		data: formData,
		processData: false,
		contentType: false,
		mimeType: "multipart/form-data",
	})
	.done(function (response) {
		console.log(response);
		document.getElementById('result').textContent = response;
	})
	.fail(function (error) {
		alert("There was an error while sending prediction request to resnet34 model."); 
		console.log(error);
	});
};

//$('#btnResNetUpload').click(uploadAndClassifyImage);