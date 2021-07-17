let app = {
    hasModel: false,
    modelId : null,
};

const MONUMENTS = ["Tour eiffel", "Arc de Triomphe", "Notre dame"];
$("#insert").on("change", function () {
    const [file] = this.files;
    const $preview = $("#preview");
    if (file) {
        $preview.attr("src", URL.createObjectURL(file));
        $('#test').removeClass('d-none');
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


function checkInputs(){
    if (app.hasModel) {
        return true;
    }
    window.alert("Vous devez d'abord charger un modèle");
    return false;
}