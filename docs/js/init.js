

(function($){

  $(function(){

    $('.sidenav').sidenav();

  }); // end of document ready


  $(document).ready(function(){
    $('.collapsible').collapsible();
  });	// Add collapse headers

document.addEventListener('DOMContentLoaded', function() {
    var elems = document.querySelectorAll('.materialboxed');
    var instances = M.Materialbox.init(elems, options);
  });


$(document).ready(function(){
    $('.materialboxed').materialbox();
  });


})(jQuery); // end of jQuery name space

