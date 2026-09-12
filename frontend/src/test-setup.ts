import "@testing-library/jest-dom/vitest";

// jsdom implements neither of these, and the image pipeline uses both.
if (!URL.createObjectURL) {
  URL.createObjectURL = () => "blob:mock";
  URL.revokeObjectURL = () => {};
}
