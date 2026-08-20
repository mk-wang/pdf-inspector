//! Preparation and OCR completion for document-level Markdown conversion.
//!
//! This module owns the smallest seam needed by native OCR adapters: the PDF is
//! detected and extracted once, OCR pages are requested as positioned tokens,
//! and one final Markdown pass assembles native and OCR text together. The
//! existing `process_pdf*` APIs use the same preparation state and therefore do
//! not maintain a second extraction pipeline.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Mutex;

use lopdf::Document;
use thiserror::Error;

use crate::detector::{self, PdfType, ScanStrategy};
use crate::markdown::MarkdownOptions;
use crate::process_mode::ProcessMode;
use crate::structure_tree::{StructRole, StructTable};
use crate::types::{LayoutComplexity, PdfLine, PdfRect, TextItem};
use crate::{
    add_ocr_reason, analyze_text_quality, compute_layout_complexity_with_chart_regions,
    detect_encoding_issues, is_garbage_text, merge_ocr_reasons, page_ocr_reasons_vec, PdfError,
    PdfOptions, PdfProcessResult, ProcessingTimer, OCR_REASON_SUSPECTED_GARBLED_TEXT,
};

/// A normalized OCR rectangle in top-left origin coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// One OCR token with page-relative layout evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrToken {
    pub text: String,
    pub bounds: NormalizedRect,
    pub line_index: u32,
    pub confidence: f32,
}

/// OCR output for one requested page.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrPageResult {
    /// 1-indexed page number.
    pub page: u32,
    pub tokens: Vec<OcrToken>,
}

/// A page that a host OCR adapter must render and recognize.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrPageRequest {
    /// 1-indexed page number.
    pub page: u32,
    pub reasons: Vec<String>,
    /// Visible page-box origin and dimensions in PDF points.
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub expected_aspect_ratio: f32,
}

/// The semantic source of a page in a completed Markdown document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridMarkdownPageSource {
    NativeText,
    Ocr,
    Blank,
}

/// Page provenance returned with a completed document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridMarkdownPage {
    /// 1-indexed page number.
    pub page: u32,
    pub source: HybridMarkdownPageSource,
    pub ocr_reasons: Vec<String>,
}

/// Complete Markdown output after every required OCR page has been submitted.
#[derive(Debug)]
pub struct HybridMarkdownDocument {
    pub markdown: String,
    pub pdf_type: PdfType,
    pub page_count: u32,
    pub pages: Vec<HybridMarkdownPage>,
    pub title: Option<String>,
    pub confidence: f32,
    pub layout: LayoutComplexity,
    pub has_encoding_issues: bool,
}

/// Errors caused by invalid OCR session use or token geometry.
#[derive(Debug, Error)]
pub enum HybridMarkdownError {
    #[error(transparent)]
    Pdf(#[from] PdfError),
    #[error("OCR page {page} was not requested")]
    UnrequestedPage { page: u32 },
    #[error("OCR page {page} was submitted more than once")]
    DuplicatePage { page: u32 },
    #[error("OCR page {page} is missing")]
    MissingPage { page: u32 },
    #[error("hybrid Markdown session is already finished")]
    SessionFinished,
    #[error("OCR token on page {page} is invalid: {reason}")]
    InvalidToken { page: u32, reason: String },
    #[error("hybrid Markdown session state is poisoned")]
    StatePoisoned,
}

#[derive(Debug, Clone, Copy)]
struct PageBox {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl PageBox {
    fn fallback() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 612.0,
            height: 792.0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct PreparedMarkdown {
    pub(crate) pdf_type: PdfType,
    pub(crate) page_count: u32,
    pub(crate) pages_needing_ocr: Vec<u32>,
    pub(crate) ocr_reasons_by_page: BTreeMap<u32, Vec<String>>,
    pub(crate) title: Option<String>,
    pub(crate) confidence: f32,
    pub(crate) source_items: Vec<TextItem>,
    pub(crate) rects: Vec<PdfRect>,
    pub(crate) lines: Vec<PdfLine>,
    pub(crate) page_thresholds: HashMap<u32, f32>,
    pub(crate) struct_roles: Option<HashMap<u32, HashMap<i64, StructRole>>>,
    pub(crate) struct_tables: Vec<StructTable>,
    pub(crate) page_filter: Option<HashSet<u32>>,
    pub(crate) markdown_options: MarkdownOptions,
    pub(crate) markdown: Option<String>,
    pub(crate) layout: LayoutComplexity,
    pub(crate) has_encoding_issues: bool,
    page_boxes: BTreeMap<u32, PageBox>,
}

#[derive(Debug)]
struct RenderedMarkdown {
    markdown: Option<String>,
    layout: LayoutComplexity,
    has_encoding_issues: bool,
    text_quality_pages: Vec<u32>,
    ocr_reasons_by_page: BTreeMap<u32, Vec<String>>,
}
struct MarkdownRenderContext<'a> {
    page_count: u32,
    page_filter: Option<&'a HashSet<u32>>,
    markdown_options: &'a MarkdownOptions,
    rects: &'a [PdfRect],
    lines: &'a [PdfLine],
    page_thresholds: &'a HashMap<u32, f32>,
    struct_roles: Option<&'a HashMap<u32, HashMap<i64, StructRole>>>,
    struct_tables: &'a [StructTable],
}

#[derive(Debug)]
struct SessionState {
    prepared: PreparedMarkdown,
    submitted: BTreeMap<u32, Vec<TextItem>>,
    finished: bool,
}

/// A resumable document-level Markdown preparation session.
///
/// Preparation owns all parser and layout state needed for one final global
/// Markdown pass. The caller submits OCR observations only for the requested
/// pages; no partial Markdown is returned.
pub struct HybridMarkdownSession {
    requests: Vec<OcrPageRequest>,
    state: Mutex<SessionState>,
}

impl std::fmt::Debug for HybridMarkdownSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HybridMarkdownSession")
            .field("requests", &self.requests)
            .finish_non_exhaustive()
    }
}

impl HybridMarkdownSession {
    pub(crate) fn from_prepared(prepared: PreparedMarkdown) -> Self {
        let requests = prepared.ocr_requests();
        Self {
            requests,
            state: Mutex::new(SessionState {
                prepared,
                submitted: BTreeMap::new(),
                finished: false,
            }),
        }
    }

    /// Return stable, ascending OCR requests.
    pub fn ocr_requests(&self) -> Vec<OcrPageRequest> {
        self.requests.clone()
    }

    /// Return the number of OCR pages already submitted.
    pub fn submitted_page_count(&self) -> Result<usize, HybridMarkdownError> {
        let state = self
            .state
            .lock()
            .map_err(|_| HybridMarkdownError::StatePoisoned)?;
        Ok(state.submitted.len())
    }

    /// Submit positioned OCR tokens for one requested page.
    pub fn submit_ocr_page(&self, result: OcrPageResult) -> Result<(), HybridMarkdownError> {
        let request = self
            .requests
            .iter()
            .find(|request| request.page == result.page)
            .ok_or(HybridMarkdownError::UnrequestedPage { page: result.page })?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| HybridMarkdownError::StatePoisoned)?;
        if state.finished {
            return Err(HybridMarkdownError::SessionFinished);
        }
        if state.submitted.contains_key(&result.page) {
            return Err(HybridMarkdownError::DuplicatePage { page: result.page });
        }
        let items = ocr_tokens_to_items(request, result.tokens)?;
        state.submitted.insert(result.page, items);
        Ok(())
    }

    /// Finish the global Markdown assembly after all OCR requests are complete.
    pub fn finish(&self) -> Result<HybridMarkdownDocument, HybridMarkdownError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| HybridMarkdownError::StatePoisoned)?;
        if state.finished {
            return Err(HybridMarkdownError::SessionFinished);
        }
        for request in &self.requests {
            if !state.submitted.contains_key(&request.page) {
                return Err(HybridMarkdownError::MissingPage { page: request.page });
            }
        }

        let mut items = state.prepared.source_items.clone();
        for (&page, ocr_items) in &state.submitted {
            items.retain(|item| item.page != page);
            items.extend(ocr_items.iter().cloned());
        }
        let rendered = state.prepared.render(items, true);
        let markdown = rendered.markdown.clone().unwrap_or_default();
        let pages = state.prepared.page_results(&state.submitted);
        state.finished = true;

        Ok(HybridMarkdownDocument {
            markdown,
            pdf_type: state.prepared.pdf_type,
            page_count: state.prepared.page_count,
            pages,
            title: state.prepared.title.clone(),
            confidence: state.prepared.confidence,
            layout: rendered.layout,
            has_encoding_issues: rendered.has_encoding_issues,
        })
    }
}

/// Prepare a complete Markdown session from an already-loaded document.
///
/// The caller retains ownership of `document`; preparation copies only the
/// extracted/layout state required for the session and does not retain the
/// `lopdf::Document`. `options.password` is ignored because the document is
/// already loaded and decrypted by the caller.
pub fn prepare_hybrid_markdown(
    document: &Document,
    mut options: PdfOptions,
) -> Result<HybridMarkdownSession, HybridMarkdownError> {
    options.mode = ProcessMode::Full;
    options.detection.strategy = ScanStrategy::Full;
    let page_count = document.get_pages().len() as u32;
    let prepared = prepare_markdown_state(document, page_count, options)?;
    Ok(HybridMarkdownSession::from_prepared(prepared))
}

pub(crate) fn process_document(
    doc: &Document,
    page_count: u32,
    options: PdfOptions,
    start: ProcessingTimer,
) -> Result<PdfProcessResult, PdfError> {
    if options.mode == ProcessMode::DetectOnly {
        let detection = detect_document(doc, page_count, &options)?;
        return Ok(PdfProcessResult {
            pdf_type: detection.pdf_type,
            markdown: None,
            page_count,
            processing_time_ms: start.elapsed_ms(),
            pages_needing_ocr: detection.pages_needing_ocr,
            ocr_reasons_by_page: page_ocr_reasons_vec(detection.ocr_reasons_by_page),
            title: detection.title,
            confidence: detection.confidence,
            layout: LayoutComplexity::default(),
            has_encoding_issues: false,
        });
    }

    let prepared = prepare_markdown_state(doc, page_count, options)?;
    Ok(PdfProcessResult {
        pdf_type: prepared.pdf_type,
        markdown: prepared.markdown,
        page_count: prepared.page_count,
        processing_time_ms: start.elapsed_ms(),
        pages_needing_ocr: prepared.pages_needing_ocr,
        ocr_reasons_by_page: page_ocr_reasons_vec(prepared.ocr_reasons_by_page),
        title: prepared.title,
        confidence: prepared.confidence,
        layout: prepared.layout,
        has_encoding_issues: prepared.has_encoding_issues,
    })
}

fn detect_document(
    doc: &Document,
    page_count: u32,
    options: &PdfOptions,
) -> Result<detector::PdfTypeResult, PdfError> {
    detector::detect_from_document_with_limit(
        doc,
        page_count,
        &options.detection,
        options.max_decompressed_size,
    )
}

fn prepare_markdown_state(
    doc: &Document,
    page_count: u32,
    options: PdfOptions,
) -> Result<PreparedMarkdown, PdfError> {
    let detection = detect_document(doc, page_count, &options)?;
    let pdf_type = detection.pdf_type;
    let mut pages_needing_ocr = detection.pages_needing_ocr;
    let title = detection.title;
    let confidence = detection.confidence;
    let detection_ocr_reasons = detection.ocr_reasons_by_page;
    let page_boxes = collect_page_boxes(doc);

    if matches!(pdf_type, PdfType::Scanned | PdfType::ImageBased) {
        return Ok(PreparedMarkdown {
            pdf_type,
            page_count,
            pages_needing_ocr,
            ocr_reasons_by_page: detection_ocr_reasons,
            title,
            confidence,
            source_items: Vec::new(),
            rects: Vec::new(),
            lines: Vec::new(),
            page_thresholds: HashMap::new(),
            struct_roles: None,
            struct_tables: Vec::new(),
            page_filter: options.page_filter,
            markdown_options: options.markdown,
            markdown: None,
            layout: LayoutComplexity::default(),
            has_encoding_issues: false,
            page_boxes,
        });
    }

    let extracted = {
        let font_cmaps =
            crate::tounicode::FontCMaps::from_doc_with_limit(doc, options.max_decompressed_size)?;
        let result = crate::extractor::extract_positioned_text_with_folio_context_with_limit(
            doc,
            &font_cmaps,
            options.page_filter.as_ref(),
            options.max_decompressed_size,
        );

        if pdf_type == PdfType::Mixed {
            match result {
                Ok(((items, rects, lines), thresholds, gid_encoded_pages)) => {
                    let sample: String = items
                        .iter()
                        .filter(|item| {
                            options
                                .page_filter
                                .as_ref()
                                .is_none_or(|filter| filter.contains(&item.page))
                        })
                        .take(200)
                        .map(|item| item.text.as_str())
                        .collect();
                    if is_garbage_text(&sample) || sample.trim().is_empty() {
                        crate::extractor::extract_positioned_text_include_invisible_with_folio_context_with_limit(
                            doc,
                            &font_cmaps,
                            options.page_filter.as_ref(),
                            options.max_decompressed_size,
                        )
                    } else {
                        Ok(((items, rects, lines), thresholds, gid_encoded_pages))
                    }
                }
                Err(error) if !error.is_decompression_limit() => {
                    crate::extractor::extract_positioned_text_include_invisible_with_folio_context_with_limit(
                        doc,
                        &font_cmaps,
                        options.page_filter.as_ref(),
                        options.max_decompressed_size,
                    )
                }
                Err(error) => Err(error),
            }
        } else {
            result
        }
    };

    let extracted = match extracted {
        Ok(extracted) => Some(extracted),
        Err(error) if pdf_type == PdfType::Mixed && !error.is_decompression_limit() => None,
        Err(error) => return Err(error),
    };

    let (struct_roles, struct_tables) = crate::structure_tree::StructTree::from_doc(doc)
        .map(|tree| {
            let page_ids = doc.get_pages();
            let roles = tree.mcid_to_roles(&page_ids);
            let tables = tree.extract_tables(&page_ids);
            let roles = if roles.is_empty() { None } else { Some(roles) };
            (roles, tables)
        })
        .unwrap_or((None, Vec::new()));

    let (source_items, rects, lines, page_thresholds, rendered, gid_pages) = match extracted {
        Some(((items, rects, lines), page_thresholds, gid_encoded_pages)) => {
            let mut ocr_reasons_by_page = BTreeMap::new();
            let unmapped_font_pages: HashSet<u32> = if pdf_type == PdfType::TextBased {
                pages_needing_ocr.iter().copied().collect()
            } else {
                HashSet::new()
            };
            let (items, rects, lines) = if unmapped_font_pages.is_empty() {
                (items, rects, lines)
            } else {
                let suppressed_pages: HashSet<u32> = unmapped_font_pages
                    .into_iter()
                    .filter(|page| {
                        options
                            .page_filter
                            .as_ref()
                            .is_none_or(|filter| filter.contains(page))
                    })
                    .collect();
                if suppressed_pages.is_empty() {
                    (items, rects, lines)
                } else {
                    for page in &suppressed_pages {
                        add_ocr_reason(
                            &mut ocr_reasons_by_page,
                            *page,
                            OCR_REASON_SUSPECTED_GARBLED_TEXT,
                        );
                    }
                    (
                        items
                            .into_iter()
                            .filter(|item| !suppressed_pages.contains(&item.page))
                            .collect(),
                        rects
                            .into_iter()
                            .filter(|rect| !suppressed_pages.contains(&rect.page))
                            .collect(),
                        lines
                            .into_iter()
                            .filter(|line| !suppressed_pages.contains(&line.page))
                            .collect(),
                    )
                }
            };
            let selected_page = |page: u32| {
                options
                    .page_filter
                    .as_ref()
                    .is_none_or(|filter| filter.contains(&page))
            };
            let rects: Vec<_> = rects
                .into_iter()
                .filter(|rect| selected_page(rect.page))
                .collect();
            let lines: Vec<_> = lines
                .into_iter()
                .filter(|line| selected_page(line.page))
                .collect();
            let gid_pages: HashSet<_> = gid_encoded_pages
                .into_iter()
                .filter(|page| selected_page(*page))
                .collect();
            let render_context = MarkdownRenderContext {
                page_count,
                page_filter: options.page_filter.as_ref(),
                markdown_options: &options.markdown,
                rects: &rects,
                lines: &lines,
                page_thresholds: &page_thresholds,
                struct_roles: struct_roles.as_ref(),
                struct_tables: &struct_tables,
            };
            let rendered = render_markdown(
                items.clone(),
                &render_context,
                ocr_reasons_by_page,
                options.mode != ProcessMode::Analyze,
            );
            (items, rects, lines, page_thresholds, rendered, gid_pages)
        }
        None => (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            HashMap::new(),
            RenderedMarkdown {
                markdown: None,
                layout: LayoutComplexity::default(),
                has_encoding_issues: false,
                text_quality_pages: Vec::new(),
                ocr_reasons_by_page: BTreeMap::new(),
            },
            HashSet::new(),
        ),
    };

    let (pdf_type, markdown, confidence) = if pdf_type == PdfType::Mixed
        && rendered
            .markdown
            .as_ref()
            .is_some_and(|markdown| is_garbage_text(markdown))
    {
        (PdfType::Scanned, None, 0.95)
    } else {
        (pdf_type, rendered.markdown, confidence)
    };

    let (markdown, has_encoding_issues, force_ocr_all) = if pdf_type == PdfType::TextBased
        && markdown
            .as_ref()
            .is_some_and(|markdown| is_garbage_text(markdown))
    {
        (None, true, true)
    } else {
        (markdown, rendered.has_encoding_issues, false)
    };

    let all_gid = !gid_pages.is_empty() && gid_pages.len() as u32 >= page_count;
    if force_ocr_all {
        pages_needing_ocr = (1..=page_count).collect();
    }
    for page in gid_pages {
        if !pages_needing_ocr.contains(&page) {
            pages_needing_ocr.push(page);
        }
    }
    for page in rendered.text_quality_pages {
        if !pages_needing_ocr.contains(&page) {
            pages_needing_ocr.push(page);
        }
    }
    pages_needing_ocr.sort_unstable();
    pages_needing_ocr.dedup();

    let mut ocr_reasons_by_page = detection_ocr_reasons;
    merge_ocr_reasons(&mut ocr_reasons_by_page, rendered.ocr_reasons_by_page);
    let markdown = if all_gid { None } else { markdown };

    Ok(PreparedMarkdown {
        pdf_type,
        page_count,
        pages_needing_ocr,
        ocr_reasons_by_page,
        title,
        confidence,
        source_items,
        rects,
        lines,
        page_thresholds,
        struct_roles,
        struct_tables,
        page_filter: options.page_filter,
        markdown_options: options.markdown,
        markdown,
        layout: rendered.layout,
        has_encoding_issues,
        page_boxes,
    })
}

fn render_markdown(
    source_items: Vec<TextItem>,
    context: &MarkdownRenderContext<'_>,
    mut ocr_reasons_by_page: BTreeMap<u32, Vec<String>>,
    emit_markdown: bool,
) -> RenderedMarkdown {
    let crate::FolioFilteredItems {
        items,
        layout_items,
        removal_mask,
        removed_pages,
    } = crate::select_items_with_document_folio_context(
        source_items,
        context.page_count,
        context.page_filter,
    );
    let text_quality = analyze_text_quality(&items);
    merge_ocr_reasons(
        &mut ocr_reasons_by_page,
        text_quality.reasons_by_page.clone(),
    );
    let chart_regions =
        crate::markdown::chart_regions_by_page(&items, context.rects, context.lines);
    let layout = compute_layout_complexity_with_chart_regions(
        &items,
        &layout_items,
        context.rects,
        context.lines,
        &chart_regions,
    );
    let markdown = if emit_markdown {
        Some(
            crate::markdown::to_markdown_from_items_with_rects_and_lines(
                items,
                context.markdown_options.clone(),
                context.rects,
                context.lines,
                crate::markdown::MarkdownDocumentContext {
                    page_thresholds: context.page_thresholds,
                    struct_roles: context.struct_roles,
                    struct_tables: context.struct_tables,
                    page_count: context.page_count,
                    prefiltered_page_number_pages: Some(&removed_pages),
                    prefiltered_page_number_mask: Some(removal_mask.as_slice()),
                    precomputed_chart_regions: Some(&chart_regions),
                },
            ),
        )
    } else {
        None
    };
    let has_encoding_issues = !ocr_reasons_by_page.is_empty()
        || text_quality.has_encoding_issues
        || markdown
            .as_ref()
            .is_some_and(|markdown| detect_encoding_issues(markdown));
    RenderedMarkdown {
        markdown,
        layout,
        has_encoding_issues,
        text_quality_pages: text_quality.pages_needing_ocr,
        ocr_reasons_by_page,
    }
}

impl PreparedMarkdown {
    fn render(&self, source_items: Vec<TextItem>, emit_markdown: bool) -> RenderedMarkdown {
        let context = MarkdownRenderContext {
            page_count: self.page_count,
            page_filter: self.page_filter.as_ref(),
            markdown_options: &self.markdown_options,
            rects: &self.rects,
            lines: &self.lines,
            page_thresholds: &self.page_thresholds,
            struct_roles: self.struct_roles.as_ref(),
            struct_tables: &self.struct_tables,
        };
        render_markdown(source_items, &context, BTreeMap::new(), emit_markdown)
    }
}
impl PreparedMarkdown {
    fn ocr_requests(&self) -> Vec<OcrPageRequest> {
        self.pages_needing_ocr
            .iter()
            .copied()
            .filter(|page| {
                self.page_filter
                    .as_ref()
                    .is_none_or(|filter| filter.contains(page))
            })
            .filter_map(|page| {
                let page_box = self
                    .page_boxes
                    .get(&page)
                    .copied()
                    .unwrap_or_else(PageBox::fallback);
                (page_box.width > 0.0 && page_box.height > 0.0).then(|| OcrPageRequest {
                    page,
                    reasons: self
                        .ocr_reasons_by_page
                        .get(&page)
                        .cloned()
                        .unwrap_or_default(),
                    x: page_box.x,
                    y: page_box.y,
                    width: page_box.width,
                    height: page_box.height,
                    expected_aspect_ratio: page_box.width / page_box.height,
                })
            })
            .collect()
    }

    fn page_results(&self, submitted: &BTreeMap<u32, Vec<TextItem>>) -> Vec<HybridMarkdownPage> {
        let pages: Vec<u32> = match &self.page_filter {
            Some(filter) => (1..=self.page_count)
                .filter(|page| filter.contains(page))
                .collect(),
            None => (1..=self.page_count).collect(),
        };
        pages
            .into_iter()
            .map(|page| {
                let source = if let Some(tokens) = submitted.get(&page) {
                    if tokens.is_empty() {
                        HybridMarkdownPageSource::Blank
                    } else {
                        HybridMarkdownPageSource::Ocr
                    }
                } else if self.source_items.iter().any(|item| item.page == page) {
                    HybridMarkdownPageSource::NativeText
                } else {
                    HybridMarkdownPageSource::Blank
                };
                HybridMarkdownPage {
                    page,
                    source,
                    ocr_reasons: self
                        .ocr_reasons_by_page
                        .get(&page)
                        .cloned()
                        .unwrap_or_default(),
                }
            })
            .collect()
    }
}

fn collect_page_boxes(doc: &Document) -> BTreeMap<u32, PageBox> {
    doc.get_pages()
        .into_iter()
        .map(|(page, page_id)| {
            let (x0, y0, x1, y1) =
                crate::extractor::page_box(doc, page_id).unwrap_or((0.0, 0.0, 612.0, 792.0));
            (
                page,
                PageBox {
                    x: x0,
                    y: y0,
                    width: (x1 - x0).max(1.0),
                    height: (y1 - y0).max(1.0),
                },
            )
        })
        .collect()
}

fn ocr_tokens_to_items(
    request: &OcrPageRequest,
    tokens: Vec<OcrToken>,
) -> Result<Vec<TextItem>, HybridMarkdownError> {
    let mut line_bounds: BTreeMap<u32, (f32, f32, f32, f32)> = BTreeMap::new();
    for token in &tokens {
        validate_ocr_token(request.page, token)?;
        if token.text.trim().is_empty() {
            continue;
        }
        let bounds = token.bounds;
        let entry = line_bounds.entry(token.line_index).or_insert((
            bounds.x,
            bounds.y,
            bounds.x + bounds.width,
            bounds.y + bounds.height,
        ));
        entry.0 = entry.0.min(bounds.x);
        entry.1 = entry.1.min(bounds.y);
        entry.2 = entry.2.max(bounds.x + bounds.width);
        entry.3 = entry.3.max(bounds.y + bounds.height);
    }

    let mut items = Vec::new();
    for token in tokens {
        if token.text.trim().is_empty() {
            continue;
        }
        let bounds = token.bounds;
        let line = line_bounds
            .get(&token.line_index)
            .copied()
            .expect("non-empty OCR token has a line bounds entry");
        let line_height = (line.3 - line.1).max(0.001);
        let font_size = (line_height * request.height).max(1.0);
        let baseline = request.y + (1.0 - line.1 - line_height) * request.height + font_size * 0.8;
        items.push(TextItem {
            text: token.text,
            x: request.x + bounds.x * request.width,
            y: baseline,
            width: (bounds.width * request.width).max(0.5),
            height: font_size,
            font: "__pdf_inspector_ocr".to_string(),
            font_size,
            page: request.page,
            is_bold: false,
            is_italic: false,
            is_underline: false,
            is_strikeout: false,
            item_type: crate::types::ItemType::Text,
            mcid: None,
        });
    }
    Ok(items)
}

fn validate_ocr_token(page: u32, token: &OcrToken) -> Result<(), HybridMarkdownError> {
    let bounds = token.bounds;
    let finite = [
        bounds.x,
        bounds.y,
        bounds.width,
        bounds.height,
        token.confidence,
    ]
    .into_iter()
    .all(f32::is_finite);
    if !finite {
        return Err(HybridMarkdownError::InvalidToken {
            page,
            reason: "geometry and confidence must be finite".to_string(),
        });
    }
    if bounds.x < 0.0
        || bounds.y < 0.0
        || bounds.width < 0.0
        || bounds.height < 0.0
        || bounds.x + bounds.width > 1.0
        || bounds.y + bounds.height > 1.0
    {
        return Err(HybridMarkdownError::InvalidToken {
            page,
            reason: "bounds must stay inside normalized 0..1 page coordinates".to_string(),
        });
    }
    if !(0.0..=1.0).contains(&token.confidence) {
        return Err(HybridMarkdownError::InvalidToken {
            page,
            reason: "confidence must be between 0 and 1".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::{content::Content, dictionary, Document, Object, Stream};

    fn text_document() -> Document {
        let mut document = Document::with_version("1.5");
        let pages_id = document.new_object_id();
        let page_id = document.new_object_id();
        let font_id = document.new_object_id();
        let content_id = document.new_object_id();
        document.objects.insert(
            font_id,
            dictionary! {
                "Type" => "Font",
                "Subtype" => "Type1",
                "BaseFont" => "Helvetica",
            }
            .into(),
        );
        let content = Content {
            operations: vec![
                lopdf::content::Operation::new("BT", vec![]),
                lopdf::content::Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), 12.into()]),
                lopdf::content::Operation::new("Td", vec![72.into(), 700.into()]),
                lopdf::content::Operation::new("Tj", vec![Object::string_literal("Native text")]),
                lopdf::content::Operation::new("ET", vec![]),
            ],
        };
        document.objects.insert(
            content_id,
            Stream::new(dictionary! {}, content.encode().unwrap()).into(),
        );
        document.objects.insert(
            page_id,
            dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
                "Resources" => dictionary! { "Font" => dictionary! { "F1" => font_id } },
                "Contents" => content_id,
            }
            .into(),
        );
        document.objects.insert(
            pages_id,
            dictionary! {
                "Type" => "Pages",
                "Kids" => vec![page_id.into()],
                "Count" => 1,
            }
            .into(),
        );
        let catalog_id = document.new_object_id();
        document.objects.insert(
            catalog_id,
            dictionary! { "Type" => "Catalog", "Pages" => pages_id }.into(),
        );
        document.trailer.set("Root", catalog_id);
        document
    }
    fn mixed_twenty_page_document() -> Document {
        let mut document = Document::with_version("1.5");
        let pages_id = document.new_object_id();
        let font_id = document.new_object_id();
        document.objects.insert(
            font_id,
            dictionary! {
                "Type" => "Font",
                "Subtype" => "Type1",
                "BaseFont" => "Helvetica",
            }
            .into(),
        );
        let image_id = document.new_object_id();
        document.objects.insert(
            image_id,
            Stream::new(
                dictionary! {
                    "Type" => "XObject",
                    "Subtype" => "Image",
                    "Width" => 1,
                    "Height" => 1,
                    "ColorSpace" => "DeviceGray",
                    "BitsPerComponent" => 8,
                },
                vec![0],
            )
            .into(),
        );

        let mut kids = Vec::new();
        for page_number in 1..=20u32 {
            let page_id = document.new_object_id();
            let content_id = document.new_object_id();
            let (resources, content) = if page_number % 2 == 1 || page_number == 20 {
                let content = Content {
                    operations: vec![
                        lopdf::content::Operation::new("BT", vec![]),
                        lopdf::content::Operation::new(
                            "Tf",
                            vec![Object::Name(b"F1".to_vec()), 12.into()],
                        ),
                        lopdf::content::Operation::new("Td", vec![72.into(), 700.into()]),
                        lopdf::content::Operation::new(
                            "Tj",
                            vec![Object::string_literal(format!("Native page {page_number}"))],
                        ),
                        lopdf::content::Operation::new(
                            "Td",
                            vec![Object::Integer(0), Object::Integer(-18)],
                        ),
                        lopdf::content::Operation::new(
                            "Tj",
                            vec![Object::string_literal("native body text")],
                        ),
                        lopdf::content::Operation::new(
                            "Td",
                            vec![Object::Integer(0), Object::Integer(-18)],
                        ),
                        lopdf::content::Operation::new(
                            "Tj",
                            vec![Object::string_literal("native footer text")],
                        ),
                        lopdf::content::Operation::new("ET", vec![]),
                    ],
                };
                (
                    dictionary! { "Font" => dictionary! { "F1" => font_id } },
                    content,
                )
            } else {
                let content = Content {
                    operations: vec![
                        lopdf::content::Operation::new("q", vec![]),
                        lopdf::content::Operation::new(
                            "cm",
                            vec![612.into(), 0.into(), 0.into(), 792.into(), 0.into(), 0.into()],
                        ),
                        lopdf::content::Operation::new("Do", vec![Object::Name(b"Im1".to_vec())]),
                        lopdf::content::Operation::new("Q", vec![]),
                    ],
                };
                (
                    dictionary! { "XObject" => dictionary! { "Im1" => image_id } },
                    content,
                )
            };
            document.objects.insert(
                content_id,
                Stream::new(dictionary! {}, content.encode().unwrap()).into(),
            );
            document.objects.insert(
                page_id,
                dictionary! {
                    "Type" => "Page",
                    "Parent" => pages_id,
                    "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
                    "Resources" => resources,
                    "Contents" => content_id,
                }
                .into(),
            );
            kids.push(page_id.into());
        }
        document.objects.insert(
            pages_id,
            dictionary! {
                "Type" => "Pages",
                "Kids" => kids,
                "Count" => 20,
            }
            .into(),
        );
        let catalog_id = document.new_object_id();
        document.objects.insert(
            catalog_id,
            dictionary! { "Type" => "Catalog", "Pages" => pages_id }.into(),
        );
        document.trailer.set("Root", catalog_id);
        document
    }

    #[test]
    fn complete_preparation_scans_every_page_for_ocr_routes() {
        let document = mixed_twenty_page_document();
        let session = prepare_hybrid_markdown(&document, PdfOptions::new()).unwrap();
        let requested: Vec<_> = session
            .ocr_requests()
            .into_iter()
            .map(|request| request.page)
            .collect();
        assert_eq!(requested, vec![2, 4, 6, 8, 10, 12, 14, 16, 18]);
    }

    #[test]
    fn session_replaces_requested_page_and_finishes_once() {
        let document = text_document();
        let session = prepare_hybrid_markdown(&document, PdfOptions::new()).unwrap();
        let requests = session.ocr_requests();
        assert!(requests.is_empty());
        let result = session.finish().unwrap();
        assert!(result.markdown.contains("Native text"));
        assert_eq!(result.pages[0].source, HybridMarkdownPageSource::NativeText);
        assert!(matches!(
            session.finish(),
            Err(HybridMarkdownError::SessionFinished)
        ));
    }

    #[test]
    fn scanned_session_merges_submitted_ocr_into_global_markdown() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/scan_with_native_header_text.pdf"
        );
        let document = Document::load(path).unwrap();
        let session = prepare_hybrid_markdown(&document, PdfOptions::new()).unwrap();
        let request = session.ocr_requests().into_iter().next().unwrap();
        assert_eq!(request.page, 1);
        assert!(request.expected_aspect_ratio > 0.0);

        assert!(matches!(
            session.finish(),
            Err(HybridMarkdownError::MissingPage { page: 1 })
        ));
        let ocr = OcrPageResult {
            page: 1,
            tokens: vec![
                OcrToken {
                    text: "Scanned".to_string(),
                    bounds: NormalizedRect {
                        x: 0.1,
                        y: 0.1,
                        width: 0.2,
                        height: 0.05,
                    },
                    line_index: 0,
                    confidence: 0.99,
                },
                OcrToken {
                    text: "content".to_string(),
                    bounds: NormalizedRect {
                        x: 0.1,
                        y: 0.2,
                        width: 0.2,
                        height: 0.05,
                    },
                    line_index: 1,
                    confidence: 0.98,
                },
            ],
        };
        assert_eq!(session.submitted_page_count().unwrap(), 0);
        session.submit_ocr_page(ocr.clone()).unwrap();
        assert_eq!(session.submitted_page_count().unwrap(), 1);
        assert!(matches!(
            session.submit_ocr_page(ocr),
            Err(HybridMarkdownError::DuplicatePage { page: 1 })
        ));

        let result = session.finish().unwrap();
        assert!(result.markdown.contains("Scanned"));
        assert!(result.markdown.contains("content"));
        assert_eq!(result.pages[0].source, HybridMarkdownPageSource::Ocr);
    }

    #[test]
    fn invalid_ocr_geometry_is_rejected_deterministically() {
        let request = OcrPageRequest {
            page: 1,
            reasons: vec!["scanned".to_string()],
            x: 0.0,
            y: 0.0,
            width: 612.0,
            height: 792.0,
            expected_aspect_ratio: 612.0 / 792.0,
        };
        let error = ocr_tokens_to_items(
            &request,
            vec![OcrToken {
                text: "bad".to_string(),
                bounds: NormalizedRect {
                    x: 0.9,
                    y: 0.0,
                    width: 0.2,
                    height: 0.1,
                },
                line_index: 0,
                confidence: 1.0,
            }],
        )
        .unwrap_err();
        assert!(matches!(error, HybridMarkdownError::InvalidToken { .. }));
    }
}
