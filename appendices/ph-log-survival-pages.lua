local first = true
function Header(header)
  if header.level == 1 then
    if first then
      first = false
    else
      return {pandoc.RawBlock("latex", "\\clearpage"), header}
    end
  end
end
