import { expect, test } from "@rstest/core";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { ArtifactLink } from "@/components/workspace/citations/artifact-link";
import { SafeStreamdown, streamdownPlugins } from "@/core/streamdown";

function renderMarkdown(content: string) {
  return renderToStaticMarkup(
    createElement(
      SafeStreamdown,
      { ...streamdownPlugins, components: { a: ArtifactLink } },
      content,
    ),
  );
}

test("adds GitHub-style heading anchors to streamdown markdown", () => {
  const html = renderMarkdown(["[概述](#概述)", "", "## 概述"].join("\n"));

  expect(html).toContain('href="#%E6%A6%82%E8%BF%B0"');
  expect(html).toContain('id="概述"');
  expect(html).not.toContain("target=");
});
