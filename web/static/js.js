let app = {
    hasModel: false,
    modelId: null,
};

const MONUMENTS = ["Tour eiffel", "Arc de Triomphe", "Notre dame"];

let $imageForm = $("#imageForm");

$imageForm.on("change", function () {
    const [file] = this.files;
    const $preview = $("#preview");
    if (file) {
        $preview.attr("src", URL.createObjectURL(file));
        $preview.removeClass('d-none');

    } else {

        console.log('rien ne va');

    }
});

$(".modelLoader").on("click", function () {
    let modelId = $(this).data("file");
    let modelType = $(this).data("modeltype");
    $.ajax({
        url: "/setModel",
        method: "POST",
        data: {
            "file": modelId,
            "type": modelType,
        },
        success: function (response, status) {
            if (status === "success") {
                window.alert("Le modèle " + modelId + " a été chargé");
                app.hasModel = true;
                app.modelId = modelId;
            }
        },
    })
});


$("#sendImage").on("click" , function (){
    if (checkInputs()){

        let form = new FormData($("form")[1]);

        console.log(form);
        $.ajax({
            url: "/predictImage",
            data: form,
            method: "POST",
            dataType: "html",
            success: function (html){
                $("#result").html(html);
            },
            processData: false, // important
            contentType: false, // important
        });
    }
})

function checkInputs() {
    if (!app.hasModel) {
        window.alert("Vous devez d'abord charger un modèle");
        return false;
    }

    if ($imageForm[0].files.length === 0) {
        window.alert("Vous devez insérer une image");
        return false;

    }
    return true;
}