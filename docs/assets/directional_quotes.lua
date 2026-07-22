-- Emit explicit directional quote macros for LaTeX instead of font-dependent
-- TeX shorthand. Pandoc does not create Str or Quoted nodes inside code spans/blocks.
local quote_macros = {
  ["“"] = "\\textquotedblleft{}",
  ["”"] = "\\textquotedblright{}",
  ["‘"] = "\\textquoteleft{}",
  ["’"] = "\\textquoteright{}",
}

function Str(element)
  if not FORMAT:match("latex") then
    return nil
  end

  local result = {}
  local cursor = 1
  local replaced = false
  while cursor <= #element.text do
    local next_start
    local next_end
    local next_macro
    for quote, macro in pairs(quote_macros) do
      local start_pos, end_pos = string.find(element.text, quote, cursor, true)
      if start_pos and (not next_start or start_pos < next_start) then
        next_start = start_pos
        next_end = end_pos
        next_macro = macro
      end
    end

    if not next_start then
      result[#result + 1] = pandoc.Str(string.sub(element.text, cursor))
      break
    end
    if next_start > cursor then
      result[#result + 1] = pandoc.Str(
        string.sub(element.text, cursor, next_start - 1)
      )
    end
    result[#result + 1] = pandoc.RawInline("latex", next_macro)
    cursor = next_end + 1
    replaced = true
  end

  if replaced then
    return result
  end
  return nil
end

function Quoted(element)
  if not FORMAT:match("latex") then
    return nil
  end

  local left
  local right
  if element.quotetype == "DoubleQuote" then
    left = "\\textquotedblleft{}"
    right = "\\textquotedblright{}"
  else
    left = "\\textquoteleft{}"
    right = "\\textquoteright{}"
  end

  local result = {pandoc.RawInline("latex", left)}
  for _, inline in ipairs(element.content) do
    result[#result + 1] = inline
  end
  result[#result + 1] = pandoc.RawInline("latex", right)
  return result
end
