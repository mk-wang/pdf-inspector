//! Public inspector entry points retain loader budget failures, including repair/password paths.
use lopdf::{Dictionary, Stream};
use pdf_inspector::{process_pdf_mem_with_options, PdfError, PdfOptions};

fn container(member: u32) -> Vec<u8> {
    let index = format!("{member} 0 ");
    let mut payload = index.clone().into_bytes();
    payload.extend_from_slice(b"<< /Pad /");
    payload.resize(61, b'A');
    payload.extend_from_slice(b" >>");
    assert_eq!(payload.len(), 64);
    let mut stream = Stream::new(Dictionary::new(), payload);
    stream.compress().unwrap();
    assert!(stream.dict.has(b"Filter"));
    let mut bytes = format!(
        "<< /Type /ObjStm /N 1 /First {} /Filter /FlateDecode /Length {} >>\nstream\n",
        index.len(),
        stream.content.len()
    )
    .into_bytes();
    bytes.extend_from_slice(&stream.content);
    bytes.extend_from_slice(b"\nendstream");
    bytes
}

fn fixture() -> Vec<u8> {
    let content = b"BT /F1 10 Tf 10 10 Td (SAFE TEXT) Tj ET";
    let mut content_body = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
    content_body.extend_from_slice(content);
    content_body.extend_from_slice(b"\nendstream");
    let bodies = [
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> /Contents 4 0 R >>".to_vec(),
        content_body,
        container(100), container(101),
    ];
    let mut pdf = b"%PDF-1.5\n".to_vec();
    let mut offsets = Vec::new();
    for (index, body) in bodies.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n", index + 1).as_bytes());
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"\nendobj\n");
    }
    let xref = pdf.len();
    pdf.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f \n", offsets.len() + 1).as_bytes(),
    );
    for offset in offsets {
        pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    pdf
}

#[test]
fn configured_limit_covers_all_containers_and_retries_preserve_its_type() {
    let input = fixture();
    let result =
        process_pdf_mem_with_options(&input, PdfOptions::detect_only().max_decompressed_size(128))
            .unwrap();
    assert_eq!(result.page_count, 1);
    for password in [None, Some("unused-on-plain-input")] {
        let mut options = PdfOptions::detect_only().max_decompressed_size(127);
        if let Some(password) = password {
            options = options.password(password);
        }
        assert!(matches!(
            process_pdf_mem_with_options(&input, options),
            Err(PdfError::DecompressionLimit { limit: 127 })
        ));
    }
}

#[test]
fn repaired_container_cannot_turn_a_limit_into_an_unrelated_parse_error() {
    let input = fixture();
    let mut wrapped = b"\xEF\xBB\xBF \n".to_vec();
    wrapped.extend_from_slice(&input);
    assert!(matches!(
        process_pdf_mem_with_options(
            &wrapped,
            PdfOptions::detect_only().max_decompressed_size(127)
        ),
        Err(PdfError::DecompressionLimit { limit: 127 })
    ));
}
