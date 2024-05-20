

const loginText = document.querySelector(".title-text .login");
const loginForm = document.querySelector("form.login");
const loginBtn = document.querySelector("label.login");
const signupBtn = document.querySelector("label.signup");
const signupLink = document.querySelector("form .signup-link a");
const toggle = document.getElementById("toggle");
const status1 = document.getElementById("status");
const bodyLogin = document.getElementById("bodyLogin");
const access_control = document.getElementsByClassName("access-control")[0];
const access_control_1 = document.getElementsByClassName("access-control")[1];
var role;

document.addEventListener('DOMContentLoaded', () => {

  
  toggle.addEventListener("change", function () {
    if (this.checked) {
      status1.textContent = "Admin";
      bodyLogin.style.background = "linear-gradient(to right, #00aeff, #41c3ff, #7fd7ff, #b6f4ff)";
      access_control.style.display = "none"
      access_control_1.style.display="none"
      status1.style.color = "#656565"
      role="authority";
    } else {
      status1.textContent = "User";
      bodyLogin.style.background = "linear-gradient(to right, #000280, #26288a, #35368e, #35368e)";
      access_control.style.display = "flex"
      access_control_1.style.display="flex"
      status1.style.color = "#fff"
      role="user"
    }
  });

signupBtn.onclick = (()=>{
  loginForm.style.marginLeft = "-50%";
  loginText.style.marginLeft = "-50%";
});
loginBtn.onclick = (()=>{
  loginForm.style.marginLeft = "0%";
  loginText.style.marginLeft = "0%";
});
signupLink.onclick = (()=>{
  signupBtn.click();
  return false;
});


$("form[name=signup_form").submit(function(e) {

    var $form = $(this);
    var $error = $form.find(".error");
    var data = $form.serialize();
    $.ajax({
      url: "/user/signup",
      type: "POST",
      data: data,
      dataType: "json",
      success: function(resp) {
        window.location.href = "/dashboard/";
      },
      error: function(resp) {
        $error.text(resp.responseJSON.error).removeClass("error--hidden");
      }
    });
  
    e.preventDefault();
  });
  $("form[name=login_form").submit(function(e) {

    var $form = $(this);
    var $error = $form.find(".error");
    var data = $form.serialize();
  
    // Add the role to the data being sent

    $.ajax({
      url: "/user/login",
      type: "POST",
      data: data,
      dataType: "json",
      success: function(resp) {
        window.location.href = "/dashboard/";
      },
      error: function(resp) {
        $error.text(resp.responseJSON.error).removeClass("error--hidden");
      }
    });
  
    e.preventDefault();
  });




});