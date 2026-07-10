export const CONFIG = {
  api: {
    topK: 5,
    timeoutMs: 30_000,
    healthPollIntervalMs: 30_000,
  },
  documents: {
    acceptedExtensions: [".pdf", ".md"],
    acceptAttr: ".pdf,.md",
    maxFileSizeMb: 20,
    uploadTimeoutMs: 120_000,
    uploadingLabel: "Ingesting document… this can take a few seconds",
    dropzoneLabel: "Drop a PDF or Markdown file here, or click to browse",
    emptyStateText: "No documents yet.",
    sectionHeading: "Documents",
  },
  input: {
    maxChars: 1000,
    charCountWarningThreshold: 500,
    maxRows: 4,
  },
  score: {
    highThreshold: 0.85,
    midThreshold: 0.7,
  },
  ui: {
    appName: "DocMind",
    sidebarTagline: "AI assistant for your technical docs",
    inputDisclaimer:
      "DocMind answers from your document corpus only. Answers may be incomplete.",
    emptyAnswerFallback:
      "DocMind returned an empty response. Try rephrasing your question.",
    emptyStateTagline: "Ask anything about your technical documents",
    scoreTooltip:
      "Retrieval relevance score — higher means this chunk was more semantically similar to your query",
  },
  exampleQuestions: [
    "What chunking strategies does DocMind support?",
    "How does hybrid retrieval differ from dense-only retrieval?",
    "What embedding models are available for ingestion?",
    "How are retrieval scores computed in Qdrant?",
  ],
} as const;
