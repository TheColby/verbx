-- Give otherwise uncaptioned Markdown tables useful titles for the List of Tables.
local current_heading = "Reference table"

function Header(element)
  local heading = pandoc.utils.stringify(element.content)
  if heading ~= "" then
    current_heading = heading
  end
end

function Table(element)
  if #element.caption.long > 0 or (element.caption.short and #element.caption.short > 0) then
    return nil
  end

  local title = current_heading
  if not string.match(string.lower(title), "table$") then
    title = title .. ": reference table"
  end
  element.caption.long = {pandoc.Plain({pandoc.Str(title)})}
  return element
end
