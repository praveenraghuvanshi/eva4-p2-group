function uploadAndClassifyImage(){
	var fileInput = document.getElementById('resnet34FileUpload').files;
	if(!fileInput.length){
		return alert('Please choose a file to upload first');
	}
	
	var file = fileInput[0];
	var filename = file.name;
	
	var formData = new FormData();
	formData.append(filename, file);
	
	console.log(filename);
	
	$.ajax({
		async: true,
		crossDomain: true,
		method: 'POST',
		url: 'https://k4n27qf9bc.execute-api.ap-south-1.amazonaws.com/dev/classify',
		data: formData,
		processData: false,
		contentType: false,
		mimeType: "multipart/form-data",
	})
	.done(function (response) {
		console.log(response);
		document.getElementById('result').textContent = response;
	})
	.fail(function () {alert("There was an error while sending prediction request to resnet34 model."); });
};

$('#btnResNetUpload').click(uploadAndClassifyImage);