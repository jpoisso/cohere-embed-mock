use actix_web::{get, HttpResponse};

#[get("/health")]
pub(crate) async fn up() -> HttpResponse {
    HttpResponse::Ok().body("up".to_string())
}
