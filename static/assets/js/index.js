const communityLink = document.getElementById("community");
const postButton = document.getElementById("postButton");
const postText = document.getElementById("postText");
const postsContainer = document.getElementById("postsContainer");
const like_button = document.getElementById("like_button")

document.addEventListener('DOMContentLoaded', function () {

  // Try


  // 
  function setActiveSection(sectionId) {
    sessionStorage.setItem('activeSection', sectionId);
  }

  function getActiveSection() {
    return sessionStorage.getItem('activeSection');
  }

  function toggleSection(sectionId) {
    const sections = ['home-content', 'community-content', 'authority-content', 'report-content'];
    sections.forEach(id => {
      if (id === sectionId) {
        document.getElementById(id).style.display = 'block';
      } else {
        document.getElementById(id).style.display = 'none';
      }
    });
  }

  // Check if there's an active section in session storage
  const activeSection = getActiveSection();
  if (activeSection) {
    toggleSection(activeSection);
  }

  const navLinks = document.querySelectorAll(".nav-link1");

  // Add click event listeners to the navigation links
  navLinks.forEach((link) => {
    link.addEventListener("click", (event) => {
      // Remove active class from all navigation links
      navLinks.forEach((navLink) => {
        navLink.classList.remove("active");
      });

      // Add active class to the clicked navigation link
      link.classList.add("active");

      // Save the ID of the clicked link in localStorage
      localStorage.setItem("activeLink", link.parentNode.id);
    });
  });

  // Check localStorage for an active link and apply active class
  const activeLinkId = localStorage.getItem("activeLink");
  if (activeLinkId) {
    const activeLink = document.getElementById(activeLinkId);
    if (activeLink) {
      activeLink.querySelector(".nav-link1").classList.add("active");
    }
  }

  // Home link click event
  document.getElementById('home').addEventListener('click', () => {
    setActiveSection('home-content');
    toggleSection('home-content');
  });

  // Community link click event
  document.getElementById('community').addEventListener('click', () => {
    setActiveSection('community-content');
    toggleSection('community-content');
  });

  document.getElementById('report').addEventListener('click', () => {
    setActiveSection('report-content');
    toggleSection('report-content');
  });
  //--------------------------------------------------------------------------------------------------

  // community codes

  // Function to append a new post to the UI

  postText.addEventListener('input', () => {
    postButton.disabled = postText.value.trim() === '';
  });


  function appendPostToUI(postData) {
    const postElement = document.createElement("div");
    postElement.className = "panel";
    const likedByUser = postData.is_liked; // Assuming currentUser_id holds the current user's ID
    console.log(likedByUser)
    const heartIconClass = likedByUser ? "fa-solid" : "fa-regular";
    const deleteButton = postData.is_owner ? '<i class="fa-solid fa-trash-can float-end" id="delete_button" data-postid="' + postData.postID + '"></i>' : '';

    postElement.innerHTML = `
          <div class="panel-body">
          <div class="media-block">
              <a class="media-left" href="#">
                  <img class="rounded-circle img-sm" alt="Profile Picture" src="https://img.freepik.com/premium-vector/business-global-economy_24877-41082.jpg?w=360">
              </a>
              
              <div class="media-body">
                  <div class="mar-btm">
                      ${deleteButton}
                      <a href="#" class="btn-link text-semibold media-heading box-inline">${postData.username}</a>
                      <p class="text-muted text-sm"><i class="fa-regular fa-clock"></i>&nbsp; ${postData.timeAgo}</p>
                      
                  </div>
                  <p>${postData.content}</p>
                  <div class="pad-ver">
                      <div class="btn-group">
                          <p p-data-postid="${postData.postID}"><i class="${heartIconClass} fa-heart" id="like_button"  data-postid="${postData.postID}"></i>${postData.like_count} Likes</p>
                          <p  class="comment-toggle" c-data-postid="${postData.postID}"><i class="fa-regular fa-comment" id="comment_button" ></i>${postData.comment_count} Comments</p>
                      </div>
                  </div>
                  <hr>
                  <div class="comments" id="comments-${postData.postID}" style="display: none;">
                      <div class="comment-list" id="comment-list-${postData.postID}"></div>
                      <div class="comment-input">
                          <input type="text" class="comment-text" placeholder="Write a comment..." data-postid="${postData.postID}">
                          <button class="comment-button" data-postid="${postData.postID}">Comment</button>
                      </div>
                  </div>
              </div>
          </div>  
      </div>
      `;

    postsContainer.appendChild(postElement);
  }

  // comment related codes
  document.addEventListener('click', async (event) => {
    const target = event.target;
    if (target.classList.contains('comment-toggle') || target.parentElement.classList.contains('comment-toggle')) {
      const postID = target.getAttribute('c-data-postid') || target.parentElement.getAttribute('c-data-postid');

      const commentsSection = document.getElementById(`comments-${postID}`);
      if (commentsSection.style.display === 'none') {
        await fetchComments(postID); // Fetch and display comments
      }
      commentsSection.style.display = commentsSection.style.display === 'none' ? 'block' : 'none';
    }
  });

  async function fetchComments(postID) {
    const commentList = document.querySelector(`#comment-list-${postID}`);
    const commentsUrl = `/get_comments_for_post/${postID}`;

    try {
      const response = await fetch(commentsUrl);
      const data = await response.json();

      commentList.innerHTML = ""; // Clear existing comments
      data.comments.forEach(comment => {
        const commentCard = `
          <div class="comment-card" style="margin-top: 20px;">
          <a class="media-left" href="#">
              <img class="rounded-circle img-sm" alt="Profile Picture" src="https://img.freepik.com/premium-vector/business-global-economy_24877-41082.jpg?w=360">
          </a>
          <div class="media-body">
              <div class="mar-btm">
                  <a href="#" class="btn-link text-semibold media-heading box-inline">${comment.user_name}</a>
                  <p class="text-muted text-sm"><i class="fa-regular fa-clock"></i>&nbsp; ${comment.timestamp}</p>
              </div>
              <p>${comment.content}</p>
          </div>
      </div>
          `;
        commentList.insertAdjacentHTML('beforeend', commentCard);
      });
    } catch (error) {
      console.error('An error occurred:', error);
    }
  }


  
  document.addEventListener('click', async (event) => {
    const target = event.target;
    if (target.classList.contains('comment-button')) {
      const postID = target.getAttribute('data-postid');
      console.log("comment", postID)
      const commentTextElement = document.querySelector(`.comment-text[data-postid="${postID}"]`);
      const commentText = commentTextElement.value.trim();

      if (commentText !== '') {
        const commentCountElement = document.querySelector(`[c-data-postid="${postID}"]`)
        const textContent = commentCountElement.textContent.trim();
        const CommentCount = parseInt(textContent.split(' ')[0]);
        const newCommentCount = CommentCount + 1;
        commentCountElement.innerHTML = `<i class="fa-regular fa-comment" id="comment_button" data-postid="${postID}"></i> ${newCommentCount} Comments`;

        try {
          const response = await fetch('/post_comment', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ postID: postID, comment: commentText }),

          });

          if (!response.ok) {
            throw new Error('Failed to post comment.');
          }

          // Clear the comment text field and update the comment list
          commentTextElement.value = '';
          await fetchComments(postID); // Refresh the comment list
        } catch (error) {
          console.error('An error occurred:', error);
          alert('An error occurred while posting the comment.');
        }
      }

      
    }
  });

  // comment ends

  // delete starts
  async function delete_post(event) {
    const target = event.target;
    if (target.id === "delete_button") {
      const postID = target.getAttribute("data-postid");
      try {
        const response = await fetch("/delete_post", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ postID: postID }),
        });

        if (response.ok) {
          // Assuming you want to refresh the posts after deleting
          await fetchPostsAndComments();
        } else {
          throw new Error("Failed to delete post.");
        }
      } catch (error) {
        console.error("An error occurred:", error);
        // Handle error
      }
    }
  }


  document.addEventListener('click', delete_post)
  // delete ends
  async function handleLikeButtonClick(event) {
    const target = event.target;
    if (target.id === 'like_button') {
      const postID = target.getAttribute("data-postid");
      console.log(postID)
      const likeCountElement = document.querySelector(`[p-data-postid="${postID}"]`);
      const heartElement = document.querySelector(`#like_button[data-postid="${postID}"]`);
      const textContent = likeCountElement.textContent.trim();
      const likeCount = parseInt(textContent.split(' ')[0]);
      const isRegular = heartElement.classList.contains('fa-regular');

      const newLikeCount = isRegular ? likeCount + 1 : likeCount - 1;
      const newIconClass = isRegular ? "fa-solid" : "fa-regular";

      likeCountElement.innerHTML = `<i class="${newIconClass} fa-heart" id="like_button" data-postid="${postID}"></i> ${newLikeCount} Likes`;
      heartElement.classList.remove("fa-regular", "fa-solid");
      heartElement.classList.add(newIconClass, "fa-heart");
      try {
        const response = await fetch("/like_post", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ postID: postID }),
        });

        if (!response.ok) {
          throw new Error("Failed to update like status.");
        }



      } catch (error) {
        console.error("An error occurred:", error);
        alert("An error have occured");
      }
    }
  }

  // Add the event listener to the document
  document.addEventListener('click', handleLikeButtonClick);


  // Function to fetch posts and comments from the server
  async function fetchPostsAndComments() {
    const response = await fetch("/get_posts_and_comments");
    const data = await response.json();
    console.log(data);
    // Clear existing posts before updating
    postsContainer.innerHTML = "";

    // Append each post to the UI
    data.posts.forEach(postData => {
      appendPostToUI(postData);
    });
  }


  // Event listener for the "Post" button
  postButton.addEventListener("click", async () => {
    const newPost = postText.value;
    const response = await fetch("/post_new", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ content: newPost }), // Send both content and post ID
    });
    if (response.ok) {
      postText.value = "";
      fetchPostsAndComments();
    }
  });



  // Event listener for the "Community" link
  communityLink.addEventListener("click", fetchPostsAndComments);

  // Initial fetch when the page loads
  fetchPostsAndComments();

  // --------------------Authority-------------------------------------
  function initializeSection() {
    const fromSlider = document.getElementById('fromSlider');
    const toSlider = document.getElementById('toSlider');
    const fromInput = document.getElementById('fromInput');
    const toInput = document.getElementById('toInput');

    // Function to update the UI based on slider values

    loadCityOptions();
    loadStateOptions();
    
    function updateUI() {
      const fromValue = parseInt(fromSlider.value);
      const toValue = parseInt(toSlider.value);

      // Update input fields
      fromInput.value = fromValue;
      toInput.value = toValue;

      // Update slider background
      const gradient = `linear-gradient(90deg, #C6C6C6 ${fromValue}%, #25daa5 ${toValue}%)`;
      fromSlider.style.background = gradient;
      toSlider.style.background = gradient;
    }

    // Event listener for the "from" slider
    fromSlider.addEventListener('input', () => {
      const fromValue = parseInt(fromSlider.value);
      const toValue = parseInt(toSlider.value);

      if (fromValue > toValue) {
        fromSlider.value = toValue;
      }

      updateUI();
    });

    // Event listener for the "to" slider
    toSlider.addEventListener('input', () => {
      const fromValue = parseInt(fromSlider.value);
      const toValue = parseInt(toSlider.value);

      if (toValue < fromValue) {
        toSlider.value = fromValue;
      }

      updateUI();
    });

    // Event listener for the "from" input
    fromInput.addEventListener('input', () => {
      const fromValue = parseInt(fromInput.value);
      const toValue = parseInt(toSlider.value);

      if (fromValue > toValue) {
        fromInput.value = toValue;
      }

      fromSlider.value = fromInput.value;
      updateUI();
    });

    // Event listener for the "to" input
    toInput.addEventListener('input', () => {
      const fromValue = parseInt(fromSlider.value);
      const toValue = parseInt(toInput.value);

      if (toValue < fromValue) {
        toInput.value = fromValue;
      }

      toSlider.value = toInput.value;
      updateUI();
    });
    function controlFromSlider(fromSlider, toSlider, fromInput) {
      const fromValue = parseInt(fromSlider.value);
      const toValue = parseInt(toSlider.value);

      if (fromValue > toValue) {
        fromSlider.value = toValue;
      }

      updateUI();
    }

    // Function to update the "to" slider and input based on input value
    function controlToSlider(fromSlider, toSlider, toInput) {
      const toValue = parseInt(toInput.value);

      if (toValue > parseInt(toSlider.max)) {
        toInput.value = toSlider.max;
      }

      toSlider.value = toInput.value;
      updateUI();
    }

    // Initial UI update
    updateUI();


    $(document).ready(function () {
      $("#rangeSlider").ionRangeSlider({
        type: "double",
        grid: true,
        min: 0,
        max: 100,
        from: 0,
        to: 55,
        step: 1,
        onFinish: function (data) {
          $("#fromInput").val(data.from);
          $("#toInput").val(data.to);
        },
      });

      $("#fromInput, #toInput").on("change", function () {
        var fromValue = parseInt($("#fromInput").val()) || 0;
        var toValue = parseInt($("#toInput").val()) || 100;
        $("#rangeSlider").data("ionRangeSlider").update({
          from: fromValue,
          to: toValue,
        });
      });
      fromSlider.oninput = () => controlFromSlider(fromSlider, toSlider, fromInput);
      toSlider.oninput = () => controlToSlider(fromSlider, toSlider, toInput);

      

      $('[data-toggle="tooltip"]').tooltip();

      $('#filter-form').submit(function (event) {
        event.preventDefault();

        var selectedMinPlasticCount = parseInt($('#fromInput').val(), 10);
        var selectedMaxPlasticCount = parseInt($('#toInput').val(), 10);
        var selectedStatus = $('input[name="radio"]:checked').val(); // Use the correct radio button selector
        var selectedCity = $('#city-filter').val();
        var selectedState = $('#state-filter').val();
        console.log("Min Plastic Count:", selectedMinPlasticCount);
        console.log("Max Plastic Count:", selectedMaxPlasticCount);
        console.log("Selected Status:", selectedStatus);

        // Send AJAX request to the backend to get filtered documents
        $.ajax({
          url: '/filter', // Update with your server route
          method: 'POST',
          data: {
            fromInput: selectedMinPlasticCount, // Use fromInput here
            toInput: selectedMaxPlasticCount,   // Use toInput here
            status: selectedStatus,
            city: selectedCity,
            state: selectedState
          },
          success: function (response) {
            console.log(response);
            $('.main-content').empty();
            if (response.length === 0) {
              var noPlasticsMessage = '<div class="alert alert-warning no-plastics-alert" role="alert">' +
                '<span class="no-plastics-icon">⚠️</span>' +
                '<span class="no-plastics-text">No plastics in your applied filter</span>' +
                '</div>'; $('.main-content').append(noPlasticsMessage);
            } else {
              response.forEach(function (document) {
                var tickIconHtml = document.status === 'pending' ?
                  '<div class="tick-icon" onclick="moveCardToCompleted(\'' + document._id + '\')">' +
                  '<i class="fas fa-check-circle" style="color: green; font-size: 24px; margin-top: 50px;"></i>' +
                  '</div>' : '';

                var cardBodyClass = document.status === 'completed' ? 'completed-card' : '';
                var cardHtml = '<div class="card mb-4 shadow-sm rounded" style="color: #007bff;" data-id="' + document._id + '">' +
                  '<div class="card-body ' + cardBodyClass + '">' +
                  '<div class="row">' +
                  '<div class="col-md-4">' +
                  '<div class="row mb-1">' +
                  '<div class="col-md-12"><strong class="font-weight-bold">City</strong></div>' +
                  '<div class="col-md-12">' + document.city + '</div>' +
                  '</div>' +
                  '<div class="row mb-1">' +
                  '<div class="col-md-12"><strong class="font-weight-bold">State</strong></div>' +
                  '<div class="col-md-12">' + document.state + '</div>' +
                  '</div>' +
                  '<div class="row mb-1">' +
                  '<div class="col-md-12"><strong class="font-weight-bold">Geotag</strong></div>' +
                  '<div class="col-md-12">' + document.geotag + '</div>' +
                  '</div>' +
                  '</div>' +
                  '<div class="col-md-4 text-left align-items-left">' +
                  '<div class="row">' +
                  '<div class="col-md-12">' +
                  '<div class="row mb-1">' +
                  '<div class="col-md-12"><strong class="font-weight-bold">Plastic Count</strong></div>' +
                  '<div class="col-md-12">' + document.plastic_count + '</div>' +
                  '</div>' +
                  '<div class="row mb-1">' +
                  '<div class="col-md-12"><strong class="font-weight-bold">Time</strong></div>' +
                  '<div class="col-md-12">' + document.time + '</div>' +
                  '</div>' +
                  '</div>' +
                  '<div class="col-md-12 mt-3">' +
                  '<a href="' +
                  document.predicted_image +
                  '" class="btn btn-link image-button" target="_blank">' +
                  '<span class="font-weight-bold">Predicted Image</span>' +
                  '<i class="fas fa-image"></i>' +
                  '</a>' +
                  '</div>' +
                  '</div>' +
                  '</div>' +
                  '<div class="col-md-4 text-center align-items-center">' +
                  tickIconHtml +
                  '<div class="mt-5">' +
                  '<div class="d-flex justify-content-center align-items-center">' +
                  '<span class="mr-3 font-weight-bold">Explainable AI</span>' +
                  '<span class="explain-icon" data-toggle="tooltip" data-placement="top" title="' + document.explain_ai_content + '">' +
                  '<i class="fas fa-info-circle"></i>' +
                  '</span>' +
                  '</div>' +
                  '</div>' +
                  '</div>' +
                  '</div>' +
                  '</div>' +
                  '</div>' +
                  '</div>'
                '<div class="modal fade" id="imageModal' + document._id + '" tabindex="-1" role="dialog" aria-labelledby="imageModalLabel" aria-hidden="true">' +
                  '<div class="modal-dialog modal-dialog-centered" role="document">' +
                  '<div class="modal-content">' +
                  '<div class="modal-header">' +
                  '<h5 class="modal-title" id="imageModalLabel">Predicted Image</h5>' +
                  '<button type="button" class="close" data-dismiss="modal" aria-label="Close">' +
                  '<span aria-hidden="true">&times;</span>' +
                  '</button>' +
                  '</div>' +
                  '<div class="modal-body text-center">' +
                  '<img src="' + document.predicted_image + '" class="img-fluid" alt="Predicted Image">' +
                  '</div>' +
                  '</div>' +
                  '</div>' +
                  '</div>';
                var imageModalHtml = '<div class="modal fade" id="imageModal' + document._id + '" tabindex="-1" role="dialog" aria-labelledby="imageModalLabel" aria-hidden="true">' +
                  '<div class="modal-dialog modal-dialog-centered" role="document">' +
                  '<div class="modal-content">' +
                  '<div class="modal-header">' +
                  '<h5 class="modal-title" id="imageModalLabel">Predicted Image</h5>' +
                  '<button type="button" class="close" data-dismiss="modal" aria-label="Close">' +
                  '<span aria-hidden="true">&times;</span>' +
                  '</button>' +
                  '</div>' +
                  '<div class="modal-body text-center">' +
                  '<img src="' + document.predicted_image + '" class="img-fluid" alt="Predicted Image">' +
                  '</div>' +
                  '</div>' +
                  '</div>' +
                  '</div>';
                $('.main-content').append(cardHtml);
              });
            }

          },

          error: function (error) {
            console.error("Error fetching filtered documents: " + error.statusText);
          }
        });
      });
    });
    function loadCityOptions() {
      // Make an AJAX request to load city options
      $.ajax({
        url: '/get_city_options', // Replace with your server route
        method: 'GET',
        success: function (response) {
          // Populate the city filter dropdown with options
          var cityFilter = $('#city-filter');
          cityFilter.empty();
          cityFilter.append('<option value="">All</option>');
          response.forEach(function (city) {
            cityFilter.append('<option value="' + city + '">' + city + '</option>');
          });
        },
        error: function (error) {
          console.error("Error loading city options: " + error.statusText);
        }
      });
    }

    function loadStateOptions() {
      // Make an AJAX request to load state options
      $.ajax({
        url: '/get_state_options', // Replace with your server route
        method: 'GET',
        success: function (response) {
          // Populate the state filter dropdown with options
          var stateFilter = $('#state-filter');
          stateFilter.empty();
          stateFilter.append('<option value="">All</option>');
          response.forEach(function (state) {
            stateFilter.append('<option value="' + state + '">' + state + '</option>');
          });
        },
        error: function (error) {
          console.error("Error loading state options: " + error.statusText);
        }
      });
    }
  }
  // 

  $(document).ready(function () {
    const authorityLink = $("#authority-link");
    const contentSection = $("#authority-content");

    authorityLink.click(function (event) {
      setActiveSection('authority-content');
      toggleSection('authority-content');
      console.log("clicked");
      event.preventDefault();

      $.ajax({
        url: '/display_documents',
        method: 'GET',
        success: function (data) {
          contentSection.html(data);
          initializeSection()
          localStorage.setItem('authorityContent', JSON.stringify(data));
        },
        error: function (error) {
          console.error(error);
        }
      });
    });

    const cachedData = localStorage.getItem('authorityContent');
    if (cachedData) {
      $.ajax({
        url: '/display_documents',
        method: 'GET',
        success: function (data) {
          contentSection.html(data);
          initializeSection()
          localStorage.setItem('authorityContent', JSON.stringify(data));
        },
        error: function (error) {
          console.error(error);
        }
      });
    }

    // Check if the current hash is "#authority" and trigger the authority link click
    if (window.location.hash === '#authority') {
      authorityLink.trigger('click');
    }
  });




});