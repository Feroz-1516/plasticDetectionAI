const mediaInput = document.getElementById('media');
const selectedFileDiv = document.getElementById('selected-file');
const predictedImage = document.getElementById('predicted-image');
const slideshowContainer = document.getElementById('slideshow-container');
const predictedImage1 = document.getElementById('predicted-image1');
const predictedImage2 = document.getElementById('predicted-image2');
const predictedImage3 = document.getElementById('predicted-image3');

const predictedVideo = document.getElementById('predicted-video');
const videoSource = document.getElementById('video-source')
const uploadButton = document.getElementById('upload-button');
const loadingMessage = document.getElementById('loading-message');
const waiting = document.getElementById('waiting');
const waiting_p = document.getElementById('waiting-p');
const processing = document.getElementById('processing');
const processing_p = document.getElementById('processing-p')
const geoloc = document.getElementsByClassName('geoloc')[0]
// ----------------------------------------------------------------------
const modalOverlay = document.getElementById('modalOverlay');
const startButton = document.getElementById('start-camera-button');//made changes
const cameraStream = document.getElementById('camera-stream');
const takePhotoButton = document.getElementById('take-photo-button');
const uploadPhotoButton = document.getElementById('upload-photo-button');//made changes
const closeButton = document.getElementById('closeModalButton');
const capturedPhoto = document.getElementById('captured-photo');
const retakePhoto = document.getElementById('retake-button');
// -------------------------------------------------------------


document.addEventListener('DOMContentLoaded', () => {
  
  //-------------------------POST COMMENTS-------------------------

//   const postComment = async () => {
//     const post_id = "YOUR_POST_ID"; // Replace with the actual post ID
//     const comment_text = document.getElementById("comment-text").value; // Get comment text

//     const requestBody = {
//         post_id,
//         comment_text
//     };

//     try {
//         const response = await fetch('/create_comment', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json'
//             },
//             body: JSON.stringify(requestBody)
//         });

//         if (!response.ok) {
//             throw new Error('Error creating comment');
//         }

//         const responseData = await response.json();
//         console.log(responseData);
//         // Refresh the comments section or perform other actions on success
//     } catch (error) {
//         console.error('Error:', error);
//         // Handle the error, show an error message, or perform other error-related actions
//     }
// };

// // Attach the event listener to the form submission
// document.getElementById("comment-form").addEventListener("submit", function (event) {
//     event.preventDefault(); // Prevent the default form submission
//     postComment(); // Call the function to post the comment
// });




  // --------------------------------------------------------------


let mediaStream;
startButton.addEventListener('click', async () => {
  console.log(modalOverlay);
  modalOverlay.style.display = 'flex';

  try {
    // mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' }
    });
    cameraStream.srcObject = mediaStream;
  } catch (error) {
    console.error('Error accessing camera:', error);
  }
});

takePhotoButton.addEventListener('click', async () => {
  const canvas = document.createElement('canvas');
  canvas.width = cameraStream.videoWidth;
  canvas.height = cameraStream.videoHeight;
  const context = canvas.getContext('2d');
  context.drawImage(cameraStream, 0, 0, canvas.width, canvas.height);

  const capturedImageURL = canvas.toDataURL('image/png');

  capturedPhoto.src = capturedImageURL;
  capturedPhoto.style.display = 'block';
  cameraStream.style.display = 'none';
  uploadPhotoButton.style.display = 'block';
  retakePhoto.style.display = 'block';
  takePhotoButton.style.display = 'none';

  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
  }
});

uploadPhotoButton.addEventListener('click', async () => {
  modalOverlay.style.display = 'none';
  slideshowContainer.style.display = "none";
  loadingMessage.style.display = 'block'; // Show loading message
  waiting.style.display = 'none';
  waiting_p.style.display = 'none';

  if (capturedPhoto.src) {
    const blob = await (await fetch(capturedPhoto.src)).blob();
    const formData = new FormData();
    formData.append('image', blob, 'captured_photo.png');

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(async (position) => {
        var latitude = position.coords.latitude;
        var longitude = position.coords.longitude;

        try {
          const locationResponse = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}`);
          const locationData = await locationResponse.json();

          if (locationData.address) {
            var address = locationData.display_name;
            var mapLink = `https://www.google.com/maps?q=${latitude},${longitude}`;

            formData.append('mapLink', mapLink);
            formData.append('address', address);
            formData.append('latitude',latitude);
            formData.append('longitude',longitude)
            const response = await fetch('/uploadPhoto', {
              method: 'POST',
              body: formData
            });

            if (response.ok) {
              const data = await response.json();
              geoloc.style.display = "block";
              var googleMapsURL = data.location;
              var googleMapsLink = document.getElementById('googleMapsLink');
              googleMapsLink.href = googleMapsURL;
              processing.style.display = "none";
              processing_p.style.display = "none";
           
              slideshowContainer.style.display = 'block';
              predictedImage.src = data.image_url;
              predictedImage1.src = data.image_url;
              predictedImage2.src = data.multi_class;
              predictedImage3.src = data.mov_stat;

              loadingMessage.style.display = 'none'; // Hide loading message
              const explanationParagraph = document.getElementById('explanation-paragraph');
              explanationParagraph.textContent = data.explanation;

            } else {
              console.error('Error:', response.statusText);
              loadingMessage.style.display = 'none'; // Hide loading message
            }
          } else {
            alert("No address found for this location.");
            loadingMessage.style.display = 'none'; // Hide loading message
          }
        } catch (error) {
          console.error("Error fetching address:", error);
          loadingMessage.style.display = 'none'; // Hide loading message
        }
      }, (error) => {
        console.error("Geolocation error:", error);
        loadingMessage.style.display = 'none'; // Hide loading message
      });
    } else {
      alert("Geolocation is not supported by this browser.");
      loadingMessage.style.display = 'none'; // Hide loading message
    }
  }
});


// Show the image in full screen when clicked
predictedImage.addEventListener("click", () => {
    if (document.fullscreenEnabled) {
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            predictedImage.requestFullscreen().catch(error => {
                console.error("Error attempting to enable full-screen mode:", error);
            });
        }
    }
});


retakePhoto.addEventListener('click', async () => {
  capturedPhoto.style.display = 'none';
  cameraStream.style.display = 'block';
  retakePhoto.style.display = 'none';
  takePhotoButton.style.display = 'block';
  uploadPhotoButton.style.display = 'none';

  try {
    // mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' }
    });
    cameraStream.srcObject = mediaStream;
    cameraContainer.style.display = 'block';
  } catch (error) {
    console.error('Error accessing camera:', error);
  }
})

closeButton.addEventListener('click', async () => {
  modalOverlay.style.display = 'none';

  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
  }
  cameraContainer.style.display = 'none';
  capturedPhoto.style.display = 'none';
  uploadPhotoButton.style.display = 'none';
});

// ----------------------address to geotag, lat, lon --------------
function generateGeotag(address, callback) {
  fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(address)}`)
      .then(response => response.json())
      .then(data => {
          if (data.length > 0) {
              var latitude = data[0].lat;
              var longitude = data[0].lon;
              var address = data[0].display_name;

              var mapLink = `https://www.google.com/maps?q=${latitude},${longitude}`;

              callback({
                  address: address,
                  latitude: latitude,
                  longitude: longitude,
                  mapLink: mapLink
              });
          } else {
              callback(null, "No location found for the provided address.");
          }
      })
      .catch(error => {
          console.error("Error fetching geocoded address:", error);
          callback(null, "Error fetching geocoded address.");
      });
}

function convertToDecimalDegrees(gpsValue) {
  if (!Array.isArray(gpsValue) || gpsValue.length !== 3) {
      return null;
  }

  const degrees = gpsValue[0].numerator / gpsValue[0].denominator;
  const minutes = gpsValue[1].numerator / gpsValue[1].denominator / 60;
  const seconds = gpsValue[2].numerator / gpsValue[2].denominator / 3600;

  return degrees + minutes + seconds;
}
// Example usage

// --------------------------------------------------------------
mediaInput.addEventListener('change', () => {
  const selectedFileName = mediaInput.files[0]?.name || 'No file chosen';
  selectedFileDiv.textContent = `Selected File: ${selectedFileName}`;
});

uploadButton.addEventListener('click', () => {
  slideshowContainer.style.display = "none";
  predictedVideo.style.display = "none";
  loadingMessage.style.display = 'block'; // Show loading message
  waiting.style.display = 'none';
  waiting_p.style.display = 'none';
  const formData = new FormData();
  formData.append('media', mediaInput.files[0]);
  imageFile = mediaInput.files[0]
  EXIF.getData(imageFile, function() {
    const exifData = EXIF.getAllTags(this);
    if (exifData.GPSLatitude && exifData.GPSLongitude) {
      // Image has geolocation data, send to Flask
      console.log("in if");
      const latitudeParts = exifData.GPSLatitude;
        const longitudeParts = exifData.GPSLongitude;
        console.log(latitudeParts)
        console.log(longitudeParts)
        // Convert the GPS latitude and longitude parts to numeric values
        const latitude = convertToDecimalDegrees(latitudeParts);
        const longitude = convertToDecimalDegrees(longitudeParts);


      const mapLink = `https://www.google.com/maps?q=${latitude},${longitude}`;
      console.log(mapLink)
      let address;
      fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}`)
        .then(response => response.json())
        .then(data => {
            if (data.display_name) {
                address = data.display_name;
                // Do something with the address
            }
        })
        .catch(error => {
            console.error('Error fetching address:', error);
        });

      
        formData.append('address',address)
        formData.append('mapLink',mapLink)
        formData.append('latitude',latitude)
        formData.append('longitude',longitude)
      fetch('/upload', {
        method: 'POST',
        body: formData
      }).then(response => response.json())
        .then(data => {
          console.log(data);
          geoloc.style.display = "block";
          var googleMapsLink = document.getElementById('googleMapsLink');
          googleMapsLink.href = mapLink;
          processing.style.display = "none";
          processing_p.style.display = "none";
          if(data.isVideo)
          {
            predictedVideo.style.display="block";
            videoSource.src = data.image_url;
          }
          else
          {
            slideshowContainer.style.display = 'block';
            predictedImage.src = data.image_url;
            predictedImage1.src= data.image_url;
            predictedImage2.src = data.multi_class;
            predictedImage3.src = data.mov_stat;

          }
          loadingMessage.style.display = 'none'; // Hide loading message
          const explanationParagraph = document.getElementById('explanation-paragraph');
          explanationParagraph.textContent = data.explanation;
    
        })
        .catch(error => {
          
          console.error('Error:', error);
          loadingMessage.style.display = 'none'; // Hide loading message
        });

    } else {
      console.log("in else");
      // Open the location modal automatically
      $('#locationModal').modal('show');

    }
  });
  

  // Get references to modal elements
  var $modal = $('#locationModal');
  var $locationInput = $('#location-input');

  // When the "Save" button is clicked
  $('#save-button').click(function () {
    var locationValue = $locationInput.val(); // Get the value of the input field
    const formData = new FormData();
    formData.append('media', mediaInput.files[0]);
    generateGeotag(locationValue, (result, error) => {
        if (error) {
            console.error(error);
            // Handle the error case if needed
        } else {
            // Use the result data to update UI and make the fetch request
            console.log('Geolocation result:', result);
            const address = result.address
            const mapLink =  result.mapLink
            const latitude =  result.latitude 
            const longitude = result.longitude

            console.log(address)
            console.log(mapLink)
            console.log(latitude)
            console.log(longitude)
            formData.append('address',address)
            formData.append('mapLink',mapLink)
            formData.append('latitude',latitude)
            formData.append('longitude',longitude)
          console.log(formData)
            // Make the fetch request to the /upload endpoint
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
              console.log(data);
              geoloc.style.display = "block";
              var googleMapsLink = document.getElementById('googleMapsLink');
              googleMapsLink.href = mapLink;
              processing.style.display = "none";
              processing_p.style.display = "none";
              if(data.isVideo)
              {
                predictedVideo.style.display="block";
                videoSource.src = data.image_url;
              }
              else
              {
                slideshowContainer.style.display = 'block';
                predictedImage.src = data.image_url;
                predictedImage1.src = data.image_url;
                predictedImage2.src = data.multi_class;
                predictedImage3.src = data.mov_stat;

              }
              loadingMessage.style.display = 'none'; // Hide loading message
              const explanationParagraph = document.getElementById('explanation-paragraph');
              explanationParagraph.textContent = data.explanation;})

            .catch(uploadError => {
                console.error('Error uploading data:', uploadError);
                // Handle the upload error if needed
            });

            // Close the modal
            $modal.modal('hide');
        }
    });
});


 $('#close-modal').click(function(){
  $modal.modal('hide');
 })
  // When the modal is hidden
  $modal.on('hidden.bs.modal', function () {
      // Clear the input field when the modal is closed
      $locationInput.val('');
  });

  
});  

});



